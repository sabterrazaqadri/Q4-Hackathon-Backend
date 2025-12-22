import typer
import asyncio
from typing import Optional
from src.services.retrieval_service import RetrievalService
from src.services.pipeline_service import PipelineService
from src.utils.validation import deterministic_validation
from src.utils.validation_dataset import get_all_test_queries, get_test_case_by_query

app = typer.Typer()


@app.command()
def query(
    query_text: str = typer.Argument(..., help="The query text to search for relevant content"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of top results to return"),
    score_threshold: Optional[float] = typer.Option(None, "--threshold", "-t", help="Minimum score threshold for results")
):
    """
    Process a query and return relevant content chunks
    """
    async def run_query():
        service = RetrievalService()
        results = await service.retrieve_relevant_chunks(
            query_text,
            top_k=top_k,
            score_threshold=score_threshold
        )

        typer.echo(f"Query: {query_text}")
        typer.echo(f"Found {len(results)} relevant chunks:")
        typer.echo("-" * 50)

        for i, chunk in enumerate(results, 1):
            typer.echo(f"{i}. Score: {chunk['score']:.4f}")
            typer.echo(f"   Content: {chunk['content'][:100]}...")
            typer.echo(f"   Source: {chunk['source_url']}")
            typer.echo(f"   Section: {chunk['section']}")
            typer.echo(f"   Module: {chunk.get('module', 'N/A')}")
            typer.echo(f"   Chapter: {chunk.get('chapter', 'N/A')}")
            typer.echo("-" * 50)

    asyncio.run(run_query())


@app.command()
def validate_pipeline():
    """
    Test the complete retrieval pipeline with known inputs
    """
    typer.echo("Validating the retrieval pipeline...")

    # Example queries for validation
    test_queries = [
        "What is ROS 2 communication?",
        "Explain URDF fundamentals",
        "How to connect AI systems to robots?"
    ]

    async def run_validation():
        service = RetrievalService()

        for query in test_queries:
            typer.echo(f"\nTesting query: '{query}'")
            results = await service.retrieve_relevant_chunks(query, top_k=3)

            if results:
                typer.echo(f"  ✓ Found {len(results)} relevant chunks")
                typer.echo(f"  First result score: {results[0]['score']:.4f}")
            else:
                typer.echo("  ✗ No results found")

    asyncio.run(run_validation())
    typer.echo("\nPipeline validation completed.")


@app.command()
def test_validation(
    tolerance: float = typer.Option(0.05, "--tolerance", "-tol", help="Tolerance for validation comparisons"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of top results to return for validation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed validation output")
):
    """
    Run comprehensive validation tests using deterministic test cases
    """
    typer.echo("Running comprehensive validation tests...")

    async def run_comprehensive_validation():
        service = RetrievalService()

        # Get all test queries
        test_queries = get_all_test_queries()
        typer.echo(f"Running {len(test_queries)} test cases...")

        all_valid = True
        total_confidence = 0.0
        detailed_results = []

        for query in test_queries:
            typer.echo(f"\nTesting: '{query}'")

            # Get expected results for this query
            test_case = get_test_case_by_query(query)
            if not test_case:
                typer.echo(f"  ✗ No test case found for query: {query}")
                all_valid = False
                continue

            expected_results = test_case["expected_results"]

            # Retrieve actual results
            actual_results = await service.retrieve_relevant_chunks(query, top_k=top_k)

            # Perform deterministic validation
            validation_result = deterministic_validation(query, expected_results, actual_results, tolerance=tolerance)

            if validation_result["is_valid"]:
                typer.echo(f"  ✓ Valid (confidence: {validation_result['confidence']:.2f})")
            else:
                typer.echo(f"  ✗ Invalid (confidence: {validation_result['confidence']:.2f})")
                all_valid = False

            total_confidence += validation_result["confidence"]

            if verbose:
                detailed_results.append({
                    "query": query,
                    "result": validation_result
                })

        avg_confidence = total_confidence / len(test_queries) if test_queries else 0
        typer.echo(f"\nOverall validation: {'PASSED' if all_valid else 'FAILED'}")
        typer.echo(f"Average confidence: {avg_confidence:.2f}")

        if verbose and detailed_results:
            typer.echo("\nDetailed validation results:")
            for result in detailed_results:
                query = result["query"]
                val_result = result["result"]
                typer.echo(f"  {query}: {val_result['confidence']:.2f}")

        return all_valid, avg_confidence

    is_valid, avg_conf = asyncio.run(run_comprehensive_validation())

    if is_valid:
        typer.echo("\n🎉 All validation tests passed!")
        raise typer.Exit(code=0)
    else:
        typer.echo("\n❌ Some validation tests failed!")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()