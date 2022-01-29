from ir_pipeline.data_transformation.get_hybrid_results import HybridSearchRunner


def test_get_hybrid_res():
    transformer = HybridSearchRunner(search_res_dir='tests/')
    transformer._ranks = [1 / (i+1) for i in range(3)]
    joined_hits = [
        ['D1', 'D2', 'D3'],
        ['D3', 'D4', 'D2'],
        ['D3', 'D2', 'D6']
    ]
    hybrid_hits = transformer._get_hybrid_res(joined_hits)
    correct_res = ['D3', 'D2', 'D1']
    assert hybrid_hits == correct_res
