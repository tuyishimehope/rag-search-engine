import argparse

from lib.semantic_search import chunk_text, embed_query_text, embed_text, search_query, verify_embeddings, verify_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    verify_parser = subparsers.add_parser("verify", help="Verify model")
    
    embed_text_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_text_parser.add_argument("text", type=str, help="Text")
    
    subparsers.add_parser("verify_embeddings", help="Verify embeddings")
    
    embed_query_parser = subparsers.add_parser("embed_query", help="Verify embeddings")
    embed_query_parser.add_argument("query", help="Search query")
    
    search_query_parser = subparsers.add_parser("search", help="search Query")
    search_query_parser.add_argument("query", type=str, help="search query")
    search_query_parser.add_argument("--limit", type=int, default=5, help="Limit")
    
    chunk_parser = subparsers.add_parser("chunk", help="text to chunk")
    chunk_parser.add_argument("text", type=str, help="Text")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="chunk size")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="overlap words")

    
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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()