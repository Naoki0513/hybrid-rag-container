# Hybrid Rag Container

このリポジトリは、Retrieval-Augmented Generation (RAG) アーキテクチャにおいて、複数の検索手法を組み合わせたハイブリッド検索を実装したコンテナイメージを作成するためのソースコードを提供します。このコンテナイメージは、以下の特徴を持っています。

1. ユーザーの質問文からキーワードを抽出し、BM25アルゴリズムを用いてキーワード検索を行うことで、関連するカテゴリーのドキュメントに絞り込みます。
2. 絞り込まれたドキュメントに対して、FAISSベクターストアを用いたベクトル検索を行い、意味的に関連性の高いドキュメントを抽出します。
3. キーワード検索とベクトル検索の結果を組み合わせることで、より精度の高い検索結果を得ることができます。
4. 抽出されたドキュメントをClaude 3 Sonnetモデルに入力し、高品質な回答文を生成します。

このハイブリッド検索アプローチにより、各検索手法の長所を活かし、短所を補完するように設計されています。

## リポジトリ構成

- `src/`: RAGアーキテクチャの実装に関連するソースコードが含まれています。
  - `rag_app.py`: メインアプリケーションファイルです。
  - `Dockerfile`: アプリケーションのコンテナイメージをビルドするためのDockerfileです。
  - `requirements.txt`: 必要なPythonパッケージのリストです。
- `data/`: RAGアーキテクチャで使用するドキュメントデータが含まれています。
  - `documents.json`: サンプルドキュメントデータです。
- `README.md`: このファイルです。リポジトリの概要を提供します。

このリポジトリのソースコードを使用して、RAGアーキテクチャにおけるハイブリッド検索を実装したコンテナイメージを作成することができます。
