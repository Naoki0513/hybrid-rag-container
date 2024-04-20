import base64
import boto3
import datetime
import json
import os
import pytz
import re
from botocore.exceptions import ClientError
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import RePhraseQueryRetriever
from langchain.schema import Document, HumanMessage
from langchain.schema.embeddings import Embeddings
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_community.chat_models import BedrockChat
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain_community.retrievers import BM25Retriever
from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler
from sudachipy import dictionary, tokenizer
from typing import List

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
BM25_TOP_K = int(os.environ["BM25_TOP_K"])
VECTORSTORE_TOP_K = int(os.environ["VECTORSTORE_TOP_K"])
LLM_MODEL_ID = os.environ["LLM_MODEL_ID"]
LLM_PROMPT = os.environ["LLM_PROMPT"]
EMBEDDING_MODEL_ID = "cohere.embed-multilingual-v3"
QUERY_GENERATOR_MODEL_ID = "anthropic.claude-instant-v1"
PARENT_CHUNK_SIZE = 20000 
CHILD_CHUNK_SIZE = 100

app = App(
	token=SLACK_BOT_TOKEN,
	signing_secret=SLACK_SIGNING_SECRET,
	process_before_response=True,
)

def generate_word_ngrams(text: str, i: int, j: int, binary: bool = False) -> List[tuple]:
	"""
	文字列を単語に分割し、指定した文字数のn-gramを生成する関数。
	"""
	try:
		print(f"Generating word n-grams for text: {text}")
		
		tokenizer_obj = dictionary.Dictionary(dict="full").create()
		mode = tokenizer.Tokenizer.SplitMode.A

		text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=20)
		texts = text_splitter.split_text(text)
		
		tokens = []
		for chunk in texts:
			chunk_tokens = tokenizer_obj.tokenize(chunk, mode)
			tokens.extend(chunk_tokens)
			
		words = [token.surface() for token in tokens]
		
		ngrams = []
		for n in range(i, j + 1):
			for k in range(len(words) - n + 1):
				ngram = tuple(words[k:k + n])
				ngrams.append(ngram)
				
		if binary:
			ngrams = list(set(ngrams))

		print(f"Generated {len(ngrams)} word n-grams")
			
		return ngrams
	
	except Exception as e:
		print(f"Error in generate_word_ngrams: {str(e)}")
		return []

def preprocess_func(text: str) -> List[str]:
	print(f"Preprocessing text: {text}")
	return generate_word_ngrams(text, 1, 1, True)

def create_retriever(texts: List[str]) -> BM25Retriever:
	"""
	BM25検索器を作成する関数。
	"""
	print("Creating BM25 retriever")
	return BM25Retriever.from_texts(texts, preprocess_func=preprocess_func, k=BM25_TOP_K)

def create_query_generator() -> LLMChain:
	"""
	質問文からキーワードを抽出するためのLLMチェーンを作成する関数。
	"""  
	print("Creating query generator LLM chain")
	
	llm = Bedrock(model_id=QUERY_GENERATOR_MODEL_ID, model_kwargs={"temperature": 0})
	prompt = PromptTemplate(
		input_variables=['question'],
		template="""
<prompt>
<task>
入力された文章から、以下の基準に従って重要な単語を抽出し、抽出された単語をスペースで区切って出力してください:
<criteria>
1. 固有名詞（人名、地名、組織名など） 
2. 専門用語や業界特有の語彙
3. 頻出する名詞や動詞（ただし、「こと」「もの」などの一般的な名詞は除く）
4. カタカナ語や外来語
5. 文章の主題に関連する語彙  
</criteria>
</task>

<example>
<input>
CodeAnalyzerは、ソースコードの静的解析ツールです。コーディング規約の遵守状況をチェックし、潜在的なバグや脆弱性を検出します。複数のプログラミング言語に対応しており、開発チームのコードの品質管理を強力にサポートします。
</input>

<output>
CodeAnalyzer ソースコード 静的解析ツール コーディング規約 遵守状況 チェック バグ 脆弱性 検出 プログラミング言語 対応 開発チーム コード品質管理 サポート
</output>  
</example>

<input>
{question}
</input>

<output>
</output>

</prompt>
"""
	)
	chain = LLMChain(llm=llm, prompt=prompt)
	return chain

def get_empty_faiss_vectorstore(embedding: Embeddings, dim: int = None, **kwargs) -> FAISS:
	print("Creating empty FAISS vector store")
	
	dummy_text, dummy_id = "1", 1
	
	if not dim:
		dummy_emb = embedding.query(dummy_text)  
	else:
		dummy_emb = [0] * dim
		
	vectorstore = FAISS.from_embeddings([(dummy_text, dummy_emb)], embedding, ids=[dummy_id], **kwargs)
	vectorstore.delete([dummy_id])
	return vectorstore

def vectorize(relevant_documents: List[Document]) -> ParentDocumentRetriever:
	"""
	BM25で絞り込まれたドキュメントをベクトル化し、FAISSベクトルストアに格納する関数。
	"""
	print("Vectorizing relevant documents")
	
	embeddings = BedrockEmbeddings(model_id=EMBEDDING_MODEL_ID)
	ID_KEY = "doc_id"
	vectorstore = get_empty_faiss_vectorstore(embeddings, 1024) 
	store = InMemoryStore()
	
	parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE)
	child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=20)

	retriever = ParentDocumentRetriever(
		vectorstore=vectorstore,
		docstore=store,
		child_splitter=child_splitter,
		parent_splitter=parent_splitter,
		id_key=ID_KEY,
		search_kwargs={"k": VECTORSTORE_TOP_K}  
	)
	ids = [str(i) for i in range(len(relevant_documents))]
	retriever.add_documents(relevant_documents, ids=ids)
	return retriever
	
def generate_llm_response(page_content: str, question: str) -> str:
	"""
	関連ドキュメントとユーザーの質問文を組み合わせてLLMにプロンプトを送信し、最終的な回答を生成する関数。
	"""
	print("Generating LLM response")
	
	prompt_text = LLM_PROMPT + "\n" + page_content
	
	llm = BedrockChat(
		model_id=LLM_MODEL_ID,
		model_kwargs={"temperature": 0}  
	)
	
	llm_response = llm.predict(text=prompt_text + "\n" + question)
	return llm_response

def get_json_from_file(file_path: str) -> dict:
	"""
	JSONファイルからデータを読み込む関数。
	"""
	print(f"Retrieving JSON data from file: {file_path}")
	
	try:
		with open(file_path, 'r', encoding='utf-8') as file:
			json_data = json.load(file)
		return json_data
	except Exception as e:
		print(f"Error in get_json_from_file: {str(e)}")
		return None

def process_question(question: str) -> str:
	"""
	一連の処理を統合し、ユーザーの質問に対する最終的な回答を生成する関数。
	"""
	print(f"Processing question: {question}")
	
	file_path = '/var/task/documents.json'
	
	print("Retrieving JSON data from file...")
	json_data = get_json_from_file(file_path)

	if json_data is None:
		return "Failed to retrieve JSON data."
	
	split_data = json_data["chunk"]
	if not split_data:
		return "Failed to split JSON data."
	
	texts = [json.dumps(item, ensure_ascii=False) for item in split_data]
	
	retriever = create_retriever(texts)
	
	query_generator = create_query_generator()
	
	rephrase_retriever = RePhraseQueryRetriever(retriever=retriever, llm_chain=query_generator)
	
	relevant_documents = rephrase_retriever.get_relevant_documents(question)
	print("Relevant documents:")
	for doc in relevant_documents:
		print(doc)
	
	print("Generated query from LLM:")
	print(rephrase_retriever.llm_chain.invoke(question))
	
	retriever = vectorize(relevant_documents)
	
	final_docs = retriever.get_relevant_documents(question)
	print("Final documents:")  
	for doc in final_docs:
		print(doc)
	
	if final_docs:
		page_content = "\n".join([doc.page_content for doc in final_docs])
		llm_response = generate_llm_response(page_content, question)
		return llm_response
	else:
		return "関連する回答が見つかりませんでした。"

def lambda_handler(event, context):
	"""
	Slackからのメンション付きの返信を受けて回答する処理を行う関数。
	"""
	print(f"Received event: {json.dumps(event)}")

	headers = event.get('headers', {})

	if 'x-slack-retry-num' in headers:
		print("Detected x-slack-retry-num. Exiting to avoid processing a retry from Slack.")
		return {
			"statusCode": 200,
			"body": json.dumps({"message": "Request identified as a retry, thus ignored."})  
		}

	if event.get('isBase64Encoded', False):
		body = base64.b64decode(event['body'])
	else:
		body = event['body']

	slack_event_json = json.loads(body)

	if "challenge" in slack_event_json:
		challenge = slack_event_json['challenge']
		return {
			"statusCode": 200,
			"headers": {"Content-Type": "application/json"},
			"body": json.dumps({"challenge": challenge})
		}

	if "event" in slack_event_json:
		slack_event = slack_event_json['event']

		if slack_event['type'] == "app_mention":
			channel = slack_event['channel']
			text = slack_event['text']
			thread_ts = slack_event.get('thread_ts', slack_event['ts'])

			text_without_mention = re.sub(r"^<@(.+?)>", "", text).strip()

			app.client.chat_postMessage(
				channel=channel,
				thread_ts=thread_ts,
				text="お問い合わせありがとうございます。回答を準備中ですので、少々お待ちください。"
			)

			final_response = process_question(text_without_mention)

			app.client.chat_postMessage(
				channel=channel,
				thread_ts=thread_ts,
				text=final_response
			)

			return {
				"statusCode": 200,
				"body": json.dumps({"message": "Success"})
			}

	return {
		"statusCode": 200,
		"body": json.dumps({"message": "Ignored"})
	}