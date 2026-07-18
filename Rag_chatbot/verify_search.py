import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('vector_retrieval', 'vector_retrieval.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
results = module.search('what is rag')
print('RESULT_COUNT', len(results))
for page, score in results[:3]:
    print(page['filename'], page['page_number'], score)
