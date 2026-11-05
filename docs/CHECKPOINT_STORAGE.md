# Checkpoint storage adapters

`providers.storage` separates integrity-checked copies to an already-mounted directory from optional remote Hub operations. Constructing an adapter does not mount a drive, authenticate, download or upload anything.

## Already-mounted directory: local, Colab Drive, Kaggle Output, Modal Volume

`DirectoryStorage` confines relative paths beneath an existing root, enforces a per-file byte ceiling and uses a `.partial` file, fsync, SHA-256 comparison and atomic rename. The default 100 GiB setting is a per-file ceiling, not a promise that the platform has that much free storage. Symlink/path traversal outside the root is rejected.

```python
from providers.storage import DirectoryStorage

storage = DirectoryStorage("/absolute/path/to/persistent-storage/minicoder", max_bytes=8 * 1024**3)
result = storage.save("run/checkpoint_step_000000008.pt",
                      "runs/tiny/checkpoints/checkpoint_step_000000008.pt")
restored = storage.restore("run/checkpoint_step_000000008.pt", "restore/checkpoint.pt",
                           expected_sha256=result["sha256"])
```

Convenience constructors are available for an already-mounted Colab Drive, Kaggle `/kaggle/working`, and a Modal Volume mounted at `/checkpoints`. The Colab helper checks the conventional Drive mountpoint; it does not mount Drive. Kaggle Output may be session-scoped until the user saves/publishes a Notebook version. A Modal Volume is persistent only when the user has provisioned and mounted it. Local disk persistence depends on the filesystem.

## Hugging Face Hub

Install the optional SDK with `pip install -r requirements-hub.txt`. Configure `HF_TOKEN` in the process environment; the code never prompts for login, reads a token from a project file, or prints the secret. Use an owner-selected repository and review its current visibility and terms.

```python
from providers.storage import HuggingFaceHubStorage

hub = HuggingFaceHubStorage("my-account/minicoder-checkpoints", repo_type="dataset")
# Network upload is blocked by default; explicitly opt in after checking repo access/visibility.
receipt = hub.save("runs/checkpoint.pt", "runs/tiny/checkpoints/checkpoint_step_000000008.pt",
                   confirm_upload=True)
# Separate download path; independently confirm before network transfer.
hub.restore("runs/checkpoint.pt", "restore/checkpoint.pt",
            expected_sha256=receipt["sha256"], confirm_download=True)
```

The Hub adapter requests a private repository when creating one and returns the commit revision plus a hash of the local source. It does not verify a remote post-upload byte-for-byte download in the upload call. Checkpoints contain pickled Python/PyTorch training state; never restore from a repository you do not trust. A checkpoint loaded by the trainer uses `torch.load(weights_only=False)` only after local sidecar SHA-256 validation, which verifies integrity relative to the sidecar—not publisher identity or safe provenance.

## Cross-runtime recovery

Cloud adapters in this repository inspect only the current process and do not remotely create sessions or move files across providers. Run training in the provider's own notebook/Studio/function, save the checkpoint into a user-mounted persistent path or explicitly approved Hub repository, then attach or download that saved artifact in the next runtime. Confirm the checkpoint manifest and SHA-256 before resuming. This handoff path is tested locally with mock files; live provider-to-provider transfers were not exercised.
