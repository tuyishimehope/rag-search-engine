import argparse
import regex as re

from lib.search_utils import load_movies
from lib.semantic_search import ChunkedSemanticSearch, chunk_text, embed_query_text, embed_text, search_query, semantic_chunking, verify_embeddings, verify_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_text_parser.add_argument("text", type=str, help="Text")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    embed_query_parser = subparsers.add_parser(
        "embed_query", help="Verify embeddings")
    embed_query_parser.add_argument("query", help="Search query")

    search_query_parser = subparsers.add_parser("search", help="search Query")
    search_query_parser.add_argument("query", type=str, help="search query")
    search_query_parser.add_argument(
        "--limit", type=int, default=5, help="Limit")

    chunk_parser = subparsers.add_parser("chunk", help="text to chunk")
    chunk_parser.add_argument("text", type=str, help="Text")
    chunk_parser.add_argument("--chunk-size", type=int,
                              default=200, help="chunk size")
    chunk_parser.add_argument("--overlap", type=int,
                              default=0, help="overlap words")

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Semantic chunk")
    semantic_chunk_parser.add_argument("text", type=str, help="Search")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4, help="--max-chunk-size")
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="--overlap")
    
    subparsers.add_parser("embed_chunks", help="embed_chunks")
    
    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search chunked")
    search_chunked_parser.add_argument("query", type=str, help="Query")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Limit")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search_query(args.query, args.limit)
        case "chunk":
            chunks = chunk_text(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")
        case "semantic_chunk":
            text = args.text
            chunks = semantic_chunking(text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")
        case "embed_chunks":
            movies = load_movies()
            chunked_semantic_search = ChunkedSemanticSearch()
            embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(
                movies)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            movies = load_movies()
            chunked_semantic_search = ChunkedSemanticSearch()
            chunked_semantic_search.load_or_create_chunk_embeddings(movies)
            results = chunked_semantic_search.search_chunks(
                args.query, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
