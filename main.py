from llama_index.readers.file import UnstructuredReader
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from pathlib import Path
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context 
from llama_index.llms.openai import OpenAI
import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

years = [2022, 2021, 2020, 2019]
loader = UnstructuredReader()



def load_all_docs() -> dict:
    doc_set = {}
    for year in years:
        year_docs = loader.load_data(
            file=Path(f"./data/UBER/UBER_{year}.html"), split_documents=False
        )
        for d in year_docs:
            d.metadata = {"year": year}
        doc_set[year] = year_docs
    return doc_set


def setup_indexes_and_store(doc_set: dict):
    index_set = {}
    for year in years:
        storage_context = StorageContext.from_defaults()
        cur_index = VectorStoreIndex.from_documents(
            doc_set[year], storage_context=storage_context
        )
        index_set[year] = cur_index
        storage_context.persist(persist_dir=f"./storage/year_{year}")
    return index_set

def load_all_stored_indexes():
    index_set = {}
    for year in years:
        storage_context = StorageContext.from_defaults(
            persist_dir=f"./storage/year_{year}"
        )
        cur_index = load_index_from_storage(storage_context)
        index_set[year] = cur_index
    return index_set


def setup_individual_query_engine_tools(index_set: dict):
    individual_query_engine_tools = [
        QueryEngineTool.from_defaults(
            query_engine=index_set[year].as_query_engine(),
            name=f"uber_{year}_query_engine",
            description=f"A query engine for the UBER {year} SEC 10-K annual report"
        )
        for year in years
    ]
    return individual_query_engine_tools


def setup_sub_question_query_engine(query_engine_tools: list)->SubQuestionQueryEngine:
    sub_question_query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools
    )
    return sub_question_query_engine

def setup_sub_query_engine_tool(sub_question_query_engine: SubQuestionQueryEngine)->QueryEngineTool:
    sub_query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=sub_question_query_engine,
        name="uber_sec_10k_query_engine",
        description=(
            "Useful for when you want to answer questions that require "
            "analysing multiple SEC 10-K documents for Uber"
        ),   
    )  
    return sub_query_engine_tool

async def main():
    load_docs_and_setup = False
    if (load_docs_and_setup):
        doc_set = load_all_docs()
        index_set = setup_indexes_and_store(doc_set)
    else:
        index_set = load_all_stored_indexes()
    individual_query_engine_tools = setup_individual_query_engine_tools(index_set)
    sub_question_query_engine = setup_sub_question_query_engine(individual_query_engine_tools)
    sub_question_query_engine_tool = setup_sub_query_engine_tool(sub_question_query_engine)
    tools = individual_query_engine_tools+[sub_question_query_engine_tool]   
    llm = OpenAI(model="gpt-4o-mini")
    agent = FunctionAgent(tools=tools, llm=llm, verbose=True)
    ctx = Context(agent)

    response = await agent.run("What is the revenue for Uber in 2022?", ctx=ctx)
    print(str(response))
    """response = await agent.run("How does it compare with 2019?", ctx=ctx)
    print(str(response))
    response = await agent.run("Compare/Contrast the risk factors described in the uber 10-k across the years. Give me answer in bullet points  ", ctx=ctx)
    print(str(response))
    """
    while True:
        user_input = input("User: ")
        if user_input == "exit":
            break   
        response = await agent.run(user_input, ctx=ctx)
        print(str(response))
if __name__ == "__main__":
    asyncio.run(main())

