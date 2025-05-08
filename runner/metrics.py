import Levenshtein


# Levenshtein similarity functions 
# This code is taken from https://github.com/andimarafioti/florence2-finetuning/blob/main/metrics.py
def normalized_levenshtein(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    distance = Levenshtein.distance(s1, s2)
    return distance / max(len_s1, len_s2)


def similarity_score(a_ij, o_q_i, tau=0.5):
    if o_q_i.endswith('.'):
        o_q_i = o_q_i[:-1]
    if a_ij.endswith('.'):
        a_ij = a_ij[:-1]
    nl = normalized_levenshtein(a_ij, o_q_i)
    return 1 - nl if nl < tau else 0

def average_normalized_levenshtein_similarity(ground_truth, predicted_answers):
    """
    Computes the average normalized Levenshtein similarity score between the ground truth and predicted answers.
    
    Args:
        ground_truth (list of str): The ground truth answers.
        predicted_answers (list of str): The predicted answers.
        
    Returns:
        float: The average normalized Levenshtein similarity score.
    """
    assert len(ground_truth) == len(
        predicted_answers
    ), "Length of ground_truth and predicted_answers must match."
    N = len(ground_truth)
    total_score = 0
    for i in range(N):
        a_i = ground_truth[i]
        o_q_i = predicted_answers[i]
        if o_q_i == "":
            # logger.warning("Skipped an empty prediction.")
            max_score = 0 
        else:
            # Handle both list and string types for ground truth (THIS LINE WAS ADDED TO HANDLE BOTH STRING AND LIST BY Sasika)
            if isinstance(a_i, list):
                max_score = max(similarity_score(a_ij.lower(), o_q_i.lower()) for a_ij in a_i)
            else:
                max_score = similarity_score(a_i.lower(), o_q_i.lower())
        total_score += max_score
    return total_score / N