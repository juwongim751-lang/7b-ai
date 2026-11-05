from pathlib import Path
import torch
from torch.utils.data import Dataset,DataLoader
from dataset.tokenizer import MiniCoderTokenizer

TEXT_EXTS={".py",".js",".jsx",".ts",".tsx",".java",".cpp",".c",".h",".hpp",".cs",".go",".rs",".php",".rb",".lua",".sql",".sh",".html",".css",".json",".md",".txt"}

def collect_text_files(root):
    root=Path(root); return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_EXTS and ".git" not in p.parts and "node_modules" not in p.parts and "venv" not in p.parts)

class MiniCoderDataset(Dataset):
    def __init__(self,files,tokenizer,max_length=1024):
        self.samples=[]; self.pad_id=tokenizer.pad_token_id or 0; self.max_length=max_length
        for path in files:
            try: text=Path(path).read_text(encoding="utf-8",errors="ignore")
            except OSError: continue
            if not text.strip(): continue
            ids=tokenizer.encode(text)
            if len(ids)<2: continue
            eos=tokenizer.eos_token_id
            if eos is not None: ids=ids+[eos]
            # Chunk long source files into next-token prediction sequences.
            for start in range(0,len(ids)-1,max_length):
                chunk=ids[start:start+max_length+1]
                if len(chunk)>=2: self.samples.append(torch.tensor(chunk,dtype=torch.long))
    def __len__(self): return len(self.samples)
    def __getitem__(self,i): return self.samples[i]

def make_collate(pad_id):
    def collate(batch):
        length=max(len(x) for x in batch); x=torch.full((len(batch),length),pad_id,dtype=torch.long); mask=torch.zeros((len(batch),length),dtype=torch.bool)
        for i,item in enumerate(batch): x[i,:len(item)]=item; mask[i,:len(item)]=True
        return x[:,:-1],x[:,1:],mask[:,:-1],mask[:,1:]
    return collate

class DatasetManager:
    def __init__(self,tokenizer): self.tokenizer=tokenizer
    def get_dataloader(self,files,batch_size=1,max_length=1024,shuffle=True,num_workers=0):
        ds=MiniCoderDataset(files,self.tokenizer,max_length)
        if not len(ds): raise ValueError("Dataset is empty. Add code/text files to the data directory.")
        return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,collate_fn=make_collate(self.tokenizer.pad_token_id or 0),pin_memory=torch.cuda.is_available())
