import datetime
import ast
import json
from SPARQLWrapper import SPARQLWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM
from utilities.timeout import time_limit, TimeoutException
import utilities.entity_link as el
import utilities.sparql_functions as sparql_f
import utilities.emb_tasks as emb_tasks
import re

# Results storage
results = dict()
results['results'] = dict()

start_time = datetime.datetime.now()
results['start_time'] = str(start_time)

sparql_wd = SPARQLWrapper("https://query.wikidata.org/sparql")

dataset_path = "../data/WikiQA-questions-sample.txt"
json_output_path = f"../output/factum_wikiqa_{start_time.timestamp()}.json"
txt_output_path = f"../output/factum_wikiqa_{start_time.timestamp()}.txt"

# Load the data
with open(dataset_path, "r", encoding='utf-8') as f:
    dataset = f.readlines()

num_correct = 0

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")
print("LLM loaded")


def evaluate_response(response, question_no, eval_no=0):
    # Extract entities from the response
    chat_ents = [{"role": "user",
                  "content": f"Extract entities from the following sentence in python list format with square brackets with no documentation:\n{response}"}]
    prompt_ents = tokenizer.apply_chat_template(chat_ents, tokenize=False, add_generation_prompt=True)
    inputs_ents = tokenizer.encode(prompt_ents, add_special_tokens=False, return_tensors="pt")
    ent_outputs = model.generate(input_ids=inputs_ents, max_new_tokens=200)
    ent_decoded = tokenizer.decode(ent_outputs[0])

    # Print the entities response
    print(f"Entities decoded: {ent_decoded}")

    # entities = ast.literal_eval(ent_decoded.split("```python\n")[1].split("\n```")[0].strip())
    entities = ast.literal_eval(re.findall(r'\[.*?\]', ent_decoded, re.DOTALL)[-1].strip())
    # Print the entities
    print(f"Entities: {entities}")
    results['results'][question_no][f'entities_{eval_no}'] = entities

    chat_trips = [{"role": "user",
                   "content": f"Given entities {entities}, extract all (subject, predicate, object entity) triples from the following sentence in table format with no documentation:\n{response}"}]
    # chat_trips = [{"role": "user",
    #                "content": f"Extract all (subject, predicate, object) facts from the following text in table format with no documentation:\n{response}"}]
    prompt_trips = tokenizer.apply_chat_template(chat_trips, tokenize=False, add_generation_prompt=True)
    inputs_trips = tokenizer.encode(prompt_trips, add_special_tokens=False, return_tensors="pt")
    trip_outputs = model.generate(input_ids=inputs_trips, max_new_tokens=200)
    trip_decoded = tokenizer.decode(trip_outputs[0])

    # Print the triples
    # print(f"Triples: {trip_decoded}")

    # Get all lines from the text with 4 | characters
    lines = trip_decoded.split("\n")
    lines = [line for line in lines if line.count("|") == 4]

    # Excluding the first two lines, which are the header and the separator, parse each line into triples
    triples = []
    for line in lines[2:]:
        triples.append(line.split("|")[1:4])

    # Remove leading and trailing whitespace from each triple
    triples = [[item.strip() for item in triple] for triple in triples]

    # Print the triples
    print(f"Triples: {triples}")
    results['results'][question_no][f'triples_{eval_no}'] = triples

    num_linked_entities = 0
    num_linked_facts = 0

    # Get the URI for each entity
    name_uri_map = dict()
    for entity in entities:
        uri = el.fetch_uri_wikidata_simple(entity)
        if uri is not None:
            name_uri_map[entity] = uri
            num_linked_entities += 1

    # Get the URI for each subject in the triples
    for triple in triples:
        if triple[0] not in name_uri_map:
            uri = el.fetch_uri_wikidata_simple(triple[0])
            if uri is not None:
                name_uri_map[triple[0]] = uri
                num_linked_entities += 1


    print(f"Linked entities: {name_uri_map}")
    results['results'][question_no][f'linked_entities_{eval_no}'] = name_uri_map

    # For each triple, perform a SPARQL query to verify the truth of the triple
    truth_scores_sum = 0
    true_facts_uris = []
    true_facts_names = []
    true_entities = dict()
    # num_linked_facts = 0

    for j, (s, p, o) in enumerate(triples):
        print(f"Triple {j + 1}: {s}, {p}, {o}")

        if s not in name_uri_map:
            continue

        s_uri = name_uri_map[s]
        uri_label_pairs = sparql_f.get_sparql_results_expanded_wikidata(s_uri)
        # print(f"URI label pairs: {uri_label_pairs}")

        # Convert SPARQL results to dictionary format
        sparql_results_dict = dict()
        for p_uri, p_label, o_uri, o_label in uri_label_pairs:
            if (p_uri, p_label) not in sparql_results_dict:
                sparql_results_dict[(p_uri, p_label)] = []
            sparql_results_dict[(p_uri, p_label)].append((o_uri, o_label))

        # Calculate similarity scores between original predicate and each predicate in the SPARQL results
        # Get a list of all predicates labels
        pred_labels = [pair[1] for pair in sparql_results_dict.keys()]
        # Calculate the similarity scores
        sim_scores = emb_tasks.calculate_cos_sim_multiple_emb(p, pred_labels)
        # Get the indices of the top 3 similarity scores
        top_sim_indices = sim_scores.argsort()[-3:][::-1]
        # Get the top 3 similarity scores
        top_sim_scores = sim_scores[top_sim_indices]
        # Get the top 3 predicate uris and labels
        top_sim_predicates = [list(sparql_results_dict.keys())[i] for i in top_sim_indices]

        print(f"Top 3 similarity scores: {top_sim_scores}")
        print(f"Top 3 predicates: {top_sim_predicates}")

        # Based on the top 3 predicates, get the most similar predicate-object pairs
        best_sim_score = 0
        best_fact_with_names = None
        best_fact_with_uris = None
        for pred_uri, pred_label in top_sim_predicates:
            for obj_uri, obj_label in sparql_results_dict[(pred_uri, pred_label)]:
                sim_score = emb_tasks.calculate_cos_sim_multiple_emb(f"{p} {o}", [f"{pred_label} {obj_label}"]).item()
                if sim_score > best_sim_score:
                    best_sim_score = sim_score
                    best_fact_with_names = (s, pred_label, obj_label)
                    best_fact_with_uris = (s_uri, pred_uri, obj_uri)

        print(f"Best fact with names: {best_fact_with_names}")
        print(f"Best fact with URIs: {best_fact_with_uris}")
        print(f"Best similarity score: {best_sim_score}")

        if best_sim_score > 0:
            truth_scores_sum += best_sim_score
            true_facts_uris.append(best_fact_with_uris)
            true_facts_names.append(best_fact_with_names)
            true_entities[best_fact_with_uris[0]] = best_fact_with_names[0]
            name_uri_map[best_fact_with_names[0]] = best_fact_with_uris[0]
            # Only add the object entity to the name_uri_map if it is not a literal
            if best_fact_with_uris[2].startswith("http"):
                true_entities[best_fact_with_uris[2]] = best_fact_with_names[2]
                name_uri_map[best_fact_with_names[2]] = best_fact_with_uris[2]

    print("Sum of Truth Scores:", truth_scores_sum)

    # Perform fuzzy evaluation normalised by number of facts extracted
    if len(triples) > 0:
        frac_true = truth_scores_sum / len(triples)
    else:
        frac_true = 0

    print("Simple measure of truthfulness:", frac_true)

    # Perform fuzzy evaluation normalised by number of linked facts
    # if num_linked_facts > 0:
    #     frac_true_linked = truth_scores_sum / num_linked_facts  # TODO: Fix this
    # else:
    #     frac_true_linked = 0

    # print("Simple measure of truthfulness (linked facts):", frac_true_linked)

    # Calculate fraction of entities linked
    # TODO: Count number of entities in both entities and name_uri_map
    num_entities = len(set(entities).union(set([triple[0] for triple in triples])))
    if len(entities) > 0:
        frac_linked_entities = num_linked_entities / num_entities  # TODO: Fix this
    else:
        frac_linked_entities = 0

    print("Fraction of linked entities:", frac_linked_entities)

    print("True facts names:", true_facts_names)
    print("True facts URIs:", true_facts_uris)

    print("True entities URIs:", true_entities.keys())
    print("True entities names:", true_entities.values())

    results['results'][question_no][f"truth_score_{eval_no}"] = truth_scores_sum
    results['results'][question_no][f"% true_{eval_no}"] = frac_true
    # results['results'][question_no][f"% true_linked_{eval_no}"] = frac_true_linked
    results['results'][question_no][f"% linked_entities_{eval_no}"] = frac_linked_entities
    results['results'][question_no][f"true_facts_names_{eval_no}"] = true_facts_names
    results['results'][question_no][f"true_facts_uris_{eval_no}"] = true_facts_uris
    results['results'][question_no][f"true_entities_uris_{eval_no}"] = list(true_entities.keys())
    results['results'][question_no][f"true_entities_names_{eval_no}"] = list(true_entities.values())

    return frac_true, frac_linked_entities, true_facts_names, true_facts_uris, true_entities, entities, name_uri_map


# Generate a response for each question
for q_no, item in enumerate(dataset):
    # Generate the initial responses
    question = item.strip()

    # Print the question
    print(f"Question {q_no+1}: {question}")

    # Generate the response
    input_ids = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**input_ids, max_new_tokens=200)
    decoded = tokenizer.decode(outputs[0])
    response = decoded.split("<bos>")[1].split("<eos>")[0][len(question)+1:].strip()

    # Print the response
    print(f"Response: {response}\n")

    results['results'][q_no] = dict()
    results['results'][q_no]['question'] = question
    results['results'][q_no]['response'] = response

    try:
        with time_limit(1500):

            frac_true, frac_linked_entities, true_facts_names, true_facts_uris, \
                true_entities, entities, name_uri_map = evaluate_response(response, q_no, 0)

            if frac_true < 0.8:
                results['results'][q_no]["below_threshold"] = True

                # Do knowledge graph enrichment
                filtered_facts = []

                # Combine true entities with entities extracted from question
                focus_entities = set(true_entities.values()).union(entities)

                # Set initial expanded facts list to true facts
                expanded_facts = true_facts_names

                if len(focus_entities) > 0:
                    # Execute SPARQL query to get the list of predicate/object pairs for each subject
                    for subject in list(focus_entities):
                        print("Subject:", subject)
                        # s_uri = el.fetch_uri_wikidata_simple(subject)
                        s_uri = name_uri_map.get(subject, None)

                        if s_uri is None:
                            continue

                        expansion = sparql_f.get_sparql_results_expanded_wikidata(s_uri)

                        # If there are no results, skip to the next subject
                        if len(expansion) == 0:
                            continue

                        expansion_dict = dict()
                        for p_uri, p_label, o_uri, o_label in expansion:
                            if (p_uri, p_label) not in expansion_dict:
                                expansion_dict[(p_uri, p_label)] = []
                            expansion_dict[(p_uri, p_label)].append((o_uri, o_label))

                        # Calculate similarity scores between response and each predicate in the SPARQL results
                        # Get a list of all predicates uris and labels
                        pred_uris_labels = list(sorted(expansion_dict.keys()))
                        pred_labels = [pair[1] for pair in pred_uris_labels]
                        # Calculate the similarity scores
                        sim_scores = emb_tasks.calculate_cos_sim_multiple_emb(response, pred_labels)
                        # Get the indices of the top 5 similarity scores (or less if there are fewer than 5)
                        top_sim_indices = sim_scores.argsort()[-5:][::-1]
                        # Get the top 5 similarity scores
                        top_sim_scores = sim_scores[top_sim_indices]
                        # Get the top 5 predicate uris and labels
                        top_sim_predicates = [pred_uris_labels[i] for i in top_sim_indices]

                        print(f"Top 5 similarity scores: {top_sim_scores}")
                        print(f"Top 5 predicates: {top_sim_predicates}")

                        # Add to expanded list of facts
                        for pred_uri, pred_label in top_sim_predicates:
                            for obj_uri, obj_label in expansion_dict[(pred_uri, pred_label)]:
                                expanded_facts.append((subject, pred_label, obj_label))

                # Convert each fact into string format
                expanded_facts_str_list = [f"{fact[0]} {fact[1]} {fact[2]}" for fact in expanded_facts]
                # print("Expanded facts:", expanded_facts_str_list)

                expanded_facts_str = ""
                if len(expanded_facts_str_list) > 0:
                    # Calculate similarity scores between question and each fact in the expanded facts
                    sim_scores_f = emb_tasks.calculate_cos_sim_multiple_emb(response, expanded_facts_str_list)
                    # Get the indices of the top 10 similarity scores
                    top_sim_indices_f = sim_scores_f.argsort()[-5:][::-1]
                    # Get the top 10 similarity scores
                    top_sim_scores_f = sim_scores_f[top_sim_indices_f]
                    # Get the top 10 expanded facts
                    top_sim_facts = [expanded_facts[i] for i in top_sim_indices_f]

                    # Convert top 10 expanded facts into string format
                    expanded_facts_str = " ".join([f"{fact[0]} {fact[1]} {fact[2]}." for fact in top_sim_facts])

                print("Expanded facts:", expanded_facts_str)
                results['results'][q_no]["expanded_facts"] = expanded_facts_str

                # Generate a new response for the question
                new_question = f"{expanded_facts_str}\nGiven the following text:\n{response}\nImprove the text based on the above context."
                input_ids = tokenizer(new_question, return_tensors="pt")
                outputs = model.generate(**input_ids, max_new_tokens=200)
                decoded = tokenizer.decode(outputs[0])
                enr_response = decoded.split("<bos>")[1].split("<eos>")[0][len(new_question)+1:].strip()

                # Print the response
                print(f"Response: {enr_response}\n")
                results['results'][q_no]['enriched_response'] = enr_response

                new_frac_true, new_frac_linked_entities, new_true_facts_names, new_true_facts_uris, \
                    new_true_entities, new_entities, new_name_uri_map = evaluate_response(enr_response, q_no, 1)

                print("Fraction of true facts improved?", new_frac_true > frac_true)
                # print("Fraction of true linked facts improved?", new_frac_true_linked > frac_true_linked)
                print("Fraction of linked entities improved?", new_frac_linked_entities > frac_linked_entities)

                # Write results to text file
                with open(txt_output_path, "a", encoding='utf-8') as f:
                    f.write(f"Question {q_no+1}\n")
                    f.write(f"Original Response: {results['results'][q_no]['response']}\n")
                    f.write(f"Original Fraction of True Facts: {frac_true}\n")
                    # f.write(f"Original Fraction of True Linked Facts: {frac_true_linked}\n")
                    f.write(f"Original Fraction of Linked Entities: {frac_linked_entities}\n")

                    f.write(f"Expanded Facts: {expanded_facts_str}\n")

                    f.write(f"Enriched Response: {results['results'][q_no]['enriched_response']}\n")
                    f.write(f"Enriched Fraction of True Facts: {new_frac_true}\n")
                    # f.write(f"Enriched Fraction of True Linked Facts: {new_frac_true_linked}\n")
                    f.write(f"Enriched Fraction of Linked Entities: {new_frac_linked_entities}\n")
                    f.write(f"Fraction of true facts improved? {new_frac_true > frac_true}\n")
                    # f.write(f"Fraction of true linked facts improved? {new_frac_true_linked > frac_true_linked}\n")
                    f.write(f"Fraction of linked entities improved? {new_frac_linked_entities > frac_linked_entities}\n\n")

                results['results'][q_no]["frac_true_better?"] = new_frac_true > frac_true
                # results['results'][q_no]["frac_true_linked_better?"] = new_frac_true_linked > frac_true_linked
                results['results'][q_no]["frac_linked_entities_better?"] = new_frac_linked_entities > frac_linked_entities

                # Save the results to json file
                with open(json_output_path, "w", encoding='utf-8') as f:
                    json.dump(results, f, indent=4)

    except TimeoutException:
        print("Timeout error")
        results['results'][q_no]['timeout'] = True
        continue

    except Exception as e:
        print("Error:", e)
        results['results'][q_no]['error'] = str(e)
        continue

# Save the results to text file
# with open(txt_output_path, "w", encoding='utf-8') as f:
#     f.write(str(results))

# Save the results to json file
with open(json_output_path, "w", encoding='utf-8') as f:
    json.dump(results, f, indent=4)
