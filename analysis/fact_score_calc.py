import json

# Load the data
with open('../output/factum_wikiqa_results.json', 'r') as file:
    results = json.load(file)

agg_init_fact_score = 0
agg_new_fact_score = 0

valid_count = 0

for result_idx in results:
    result = results[result_idx]
    if '% true_0' in result:
        init_fact_score = result['% true_0']
        new_fact_score = result['% true_1']

        agg_init_fact_score += init_fact_score
        agg_new_fact_score += new_fact_score

        valid_count += 1

print(f'Average initial fact score: {agg_init_fact_score / valid_count}')
print(f'Average new fact score: {agg_new_fact_score / valid_count}')
print(f'Valid count: {valid_count}')

