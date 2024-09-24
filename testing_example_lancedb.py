from gptcache import cache
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from langchain_openai import OpenAIEmbeddings


from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_community.vectorstores import LanceDB
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from gptcache.adapter.langchain_models import LangChainLLMs



# get the content(only question) form the prompt to cache
def get_content_func(data, **_):
    return data.get("prompt").split("Question")[-1]


onnx = Onnx()
cache_base = CacheBase('sqlite')
vector_base = VectorBase('lancedb')
data_manager = get_data_manager(cache_base, vector_base)
cache.init(
    pre_embedding_func=get_content_func,
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    )
cache.set_openai_key()

print("cache initilised for lancedb")




# from langchain_chroma import Chroma


loader = TextLoader('/home/akashdesai/projects/lancedb/gptchache_dev/GPTCache/state_of_the_union.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


print("docs loaded")

loader = TextLoader('state_of_the_union.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key='sk-')
vector_db = LanceDB.from_documents(
    docs,
    embeddings,
)
query = "What did the president say about Justice Breyer"
docs = vector_db.similarity_search(query)



# llm = OpenAI(temperature=0) # using the following code to cache with gptcache
llm = LangChainLLMs(llm=OpenAI(temperature=0,openai_api_key='sk-'))
chain = load_qa_chain(llm, chain_type="stuff")
query = "What did the president say about Justice Breyer"
op =chain.run(input_documents=docs, question=query)
print("complet")
print("op",op)

