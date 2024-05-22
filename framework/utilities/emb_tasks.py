import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
# from angle_emb import AnglE, Prompts

model = SentenceTransformer("all-MiniLM-L6-v2")

def calculate_cos_sim(text1, text2):
    text1_emb_opt = openai.Embedding.create(
        input=text1,
        model="text-embedding-ada-002"
    )

    text1_embeddings = text1_emb_opt['data'][0]['embedding']

    text2_emb_opt = openai.Embedding.create(
        input=text2,
        model="text-embedding-ada-002"
    )

    text2_embeddings = text2_emb_opt['data'][0]['embedding']

    # Calculate the cosine similarity between the LLM response and the predicate
    cos_sim = cosine_similarity([text1_embeddings], [text2_embeddings])

    return np.square(cos_sim[0][0])


def calculate_squared_cos_sim(text1, text2):
    return np.square(calculate_cos_sim(text1, text2))


def calculate_cos_sim_multiple(text1, texts_list):
    text1_emb_opt = openai.Embedding.create(
        input=text1,
        model="text-embedding-ada-002"
    )

    text1_embeddings = text1_emb_opt['data'][0]['embedding']

    embs_list = []

    for text in texts_list:
        text_emb_opt = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )

        text_embeddings = text_emb_opt['data'][0]['embedding']
        embs_list.append(text_embeddings)

    # Calculate the cosine similarity between the LLM response and the predicate
    cos_sims = cosine_similarity([text1_embeddings], embs_list)

    return np.array(cos_sims[0])


def calculate_squared_cos_sim_multiple(text1, texts_list):
    return np.square(calculate_cos_sim_multiple(text1, texts_list))


def calculate_cos_sim_multiple_emb(text1, texts_list, print_shapes=False):
    vec = model.encode([text1])
    vecs = model.encode(texts_list)

    # if print_shapes:
    #     print("Shape of vec:", vec.shape)
    #     print("Shape of vecs:", vecs.shape)

    return cosine_similarity(vec, vecs)[0]

# def calculate_multiple_cos_sims_uae(text1, texts_list):
#     angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
#     angle.set_prompt(prompt=Prompts.C)
#     vec = angle.encode({'text': text1}, to_numpy=True)
#     vecs = angle.encode([{'text': text} for text in texts_list], to_numpy=True)
#
#     print("Shape of vec:", vec.shape)
#     print("Shape of vecs:", vecs.shape)
#
#     return cosine_similarity(vec, vecs)[0]

# print(calculate_squared_cos_sim_multiple("Paris", ["France", "London", "Berlin", "Bordeaux"]))
# print(calculate_squared_cos_sim("Subsistence economies", "The Glass Menagerie"))

# print(calculate_multiple_cos_sims_uae("Paris", ["France", "London", "Berlin", "Bordeaux"]))
