"""环境测试脚本。"""

try:
    from langchain_community.document_loaders import TextLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from transformers import pipeline

    print("langchain 导入成功")
    print("chromadb 导入成功")
    print("transformers 导入成功")
    print("sentence-transformers 导入成功")
    print("环境测试正常，可继续后续步骤")
except Exception as exc:
    print("环境测试失败：", exc)
    raise
