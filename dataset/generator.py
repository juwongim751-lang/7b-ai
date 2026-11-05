import random
import re
from typing import List, Dict

class CodeGenerator:
    """
    합성 코딩 데이터를 생성하는 클래스.
    실제 대규모 학습 전, 데이터 파이프라인 테스트를 위해 구조화된 예제를 생성합니다.
    """
    def __init__(self):
        self.languages = {
            "python": self._gen_python,
            "javascript": self._gen_js,
            "cpp": self._gen_cpp,
            "sql": self._gen_sql,
            "lua": self._gen_lua
        }

    def _gen_python(self):
        templates = [
            "def {func_name}({args}):\n    {body}\n    return {ret}",
            "class {class_name}:\n    def __init__(self, {args}):\n        {body}",
            "if __name__ == '__main__':\n    {body}"
        ]
        # 단순 예제 데이터 생성 로직 (실제로는 더 복잡한 문법 트리 기반 생성기 사용)
        return random.choice(templates).format(
            func_name=random.choice(["calculate_sum", "get_user_data", "process_request"]),
            args=random.choice(["a, b", "data", "request, response"]),
            body="print('Processing...')",
            ret="None",
            class_name=random.choice(["UserManager", "DataProcessor", "NetworkClient"])
        )

    def _gen_js(self):
        return f"function {random.choice(['fetchData', 'updateUI'])}(url) {{\n  return fetch(url).then(res => res.json());\n}}"

    def _gen_cpp(self):
        return f"#include <iostream>\nint main() {{\n    std::cout << 'Hello' << std::endl;\n    return 0;\n}}"

    def _gen_sql(self):
        return f"SELECT * FROM {random.choice(['users', 'orders', 'products'])} WHERE id = 1;"

    def _gen_lua(self):
        return f"function greet(name)\n    print('Hello ' .. name)\nend"

    def generate_batch(self, size=1000) -> List[Dict]:
        dataset = []
        for _ in range(size):
            lang = random.choice(list(self.languages.keys()))
            code = self.languages[lang]()
            dataset.append({"language": lang, "code": code})
        return dataset

class CodeValidator:
    """
    생성된 코드의 품질을 검증하는 클래스.
    """
    @staticmethod
    def is_valid_syntax(code: str, lang: str) -> bool:
        # 실제 구현에서는 각 언어의 AST 파서(ast.parse 등)를 사용합니다.
        if not code or len(code) < 5:
            return False
        return True

    @staticmethod
    def deduplicate(dataset: List[Dict]) -> List[Dict]:
        seen = set()
        unique_data = []
        for item in dataset:
            code_hash = hash(item['code'])
            if code_hash not in seen:
                seen.add(code_hash)
                unique_data.append(item)
        return unique_data

if __name__ == "__main__":
    gen = CodeGenerator()
    val = CodeValidator()
    
    raw_data = gen.generate_batch(100)
    clean_data = val.deduplicate([item for item in raw_data if val.is_valid_syntax(item['code'], item['language'])])
    
    print(f"Generated: {len(raw_data)}, Cleaned: {len(clean_data)}")
    print(f"Sample: {clean_data[0]['code']}")
