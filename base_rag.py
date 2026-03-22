"""基础 RAG 脚本：加载机械数据、检索、生成回答。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from chroma_compat import get_chroma_client_settings
from llm_loader import get_llm
from model_provider import resolve_model_path

DEFAULT_MODEL = "Qwen/Qwen2-0.5B-Instruct"
DEFAULT_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DB_DIR = "./chroma_db"


def build_chain(data_path: str, model_name: str, embedding_model_name: str, db_dir: str) -> RetrievalQA:
    loader = TextLoader(data_path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    splits = splitter.split_documents(documents)

    embedding_model_path = resolve_model_path(embedding_model_name)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
    vector_db = Chroma.from_documents(
        splits,
        embeddings,
        persist_directory=db_dir,
        client_settings=get_chroma_client_settings(db_dir),
    )
    vector_db.persist()

    llm = get_llm(model_name=model_name, max_new_tokens=256)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
    )


def run_demo(chain: RetrievalQA, questions: List[str]) -> None:
    print("=" * 50)
    print("基础 RAG 测试结果")
    print("=" * 50)
    for idx, question in enumerate(questions, 1):
        result = chain.invoke({"query": question})
        print(f"\n第{idx}个问题：{question}")
        print(f"RAG 响应：{result['result']}")
        print("检索到的源数据：")
        for doc in result["source_documents"]:
            print("-", doc.page_content[:120])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="mechanical_data.txt")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--embedding_model_name", default=DEFAULT_EMBEDDING)
    parser.add_argument("--db_dir", default=DEFAULT_DB_DIR)
    args = parser.parse_args()

    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"未找到数据文件: {args.data_path}")

    questions = [
        "机械臂夹爪的力范围是多少？",
        "机械臂标定的流程是什么？",
        "抓取光滑金属零件时，夹爪力应控制在什么范围？",
        "具身智能机械臂的RAG检索需要提供哪些机械相关数据？",
    ]

    chain = build_chain(args.data_path, args.model_name, args.embedding_model_name, args.db_dir)
    run_demo(chain, questions)
    print("\n基础 RAG 搭建完成。")


if __name__ == "__main__":
    main()
