import re
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load a small semantic model (run once, keep as global variable)
st_model = SentenceTransformer('all-MiniLM-L6-v2')

valid_pattern = re.compile(r'^(search|click|think)\[.*\]$')

def xyz_function(responses):
    """
    Takes list of responses from LLM and returns single best action.
    Following the algorithm:
    1. Filter invalid actions using regex strict matching
    2. Action Type Classification & Grouping (search, click, think)
    3. Majority Voting Selection
    4. Semantic Similarity Clustering (for search queries & reasoning)
    5. Return best action
    """
    if not responses:
        return "search[fallback]"

    # # Step 1: Clean and filter raw responses to valid actions only
    # filtered_responses = []
    # for response in responses:
    #     cleaned = response.strip() #removes leading and trailing whitespace (like spaces, newlines, tabs).
    #     if cleaned.startswith('Action: '): #checks if the string begins with that exact text. If true, cleaned = cleaned[8:] slices off the first 8 characters ("Action: ) so only the actual action remains.
    #         cleaned = cleaned[8:] #Input: "Action: search[gluten free bacon]" After slicing: "search[gluten free bacon]"
    #     if valid_pattern.match(cleaned): #Ensures the format is exactly like search[...], click[...], or think[...].
    #         filtered_responses.append(cleaned)

    # if not filtered_responses:
    #     return "search[error]"

    # print(f"Filtered valid responses count: {len(filtered_responses)}")

    # Use the filtered list of valid actions from the previous step
    valid_actions = responses

    # Create a dictionary to group actions into three categories:
    # SEARCH, CLICK, THINK
    action_groups = {
        'SEARCH': [],   # will hold all search[...] actions
        'CLICK': [],    # will hold all click[...] actions
        'THINK': []     # will hold all think[...] actions
    }

    # Iterate over each valid action
    for action in valid_actions:
        # If the action starts with "search[", put it into the SEARCH bucket
        if action.startswith('search['):
            action_groups['SEARCH'].append(action)

        # If the action starts with "click[", put it into the CLICK bucket
        elif action.startswith('click['):
            action_groups['CLICK'].append(action)

        # If the action starts with "think[", put it into the THINK bucket
        elif action.startswith('think['):
            action_groups['THINK'].append(action)

    # Step 3: Identify majority action type using majority voting
    group_counts = {k: len(v) for k, v in action_groups.items() if v} # Count how many actions are in each group, ignoring empty groups
    if not group_counts:
        return valid_actions[0]

    majority_type = max(group_counts, key=group_counts.get) # Find the group (SEARCH, CLICK, or THINK) with the maximum count
    majority_actions = action_groups[majority_type] # Get the actual list of actions for that majority type

    print(f"Action type distribution: {group_counts}")
    print(f"Majority type: {majority_type} with {len(majority_actions)} actions")

    # Step 4: Selection strategy based on action type
    if majority_type == 'SEARCH':
        # Apply semantic clustering for search queries
        return select_best_search_query(majority_actions)
    elif majority_type == 'CLICK':
        # Apply majority voting for click actions
        return select_by_majority_voting(majority_actions)
    elif majority_type == 'THINK':
        # Select best reasoning action
        return select_best_reasoning(majority_actions)
    else:
        return majority_actions[0]

def select_best_search_query(search_actions):
    """
    Choose the most representative `search[...]` action by:
      1. extracting the query text inside brackets,
      2. embedding all queries,
      3. computing the centroid (mean embedding),
      4. returning the original action whose embedding is closest to centroid by cosine similarity.
    """
    if len(search_actions) == 1:
        return search_actions[0]

    # Safely extract query text and normalize whitespace.
    # We use .strip() to remove accidental leading/trailing spaces inside brackets.
    queries = [action[7:-1] for action in search_actions]  # Everything between search[...]

    # Encode queries into dense vectors (NumPy by default).
    embeddings = st_model.encode(queries)

    # Compute the centroid (mean vector) along the row axis (mean of N vectors -> single D-dim vector)
    centroid = np.mean(embeddings, axis=0)
 
    # Compute cosine similarity between each embedding and the centroid.
    # util.cos_sim supports arrays/tensors; this yields an (N, 1) or (N,) result.
    cos_scores = util.cos_sim(embeddings, centroid)

    # Get index of highest cosine similarity (most similar to centroid).
    # np.argmax flattens if needed; wrap in int() to ensure a plain Python int.
    best_idx = int(np.argmax(cos_scores))

    
    best_query_action = search_actions[best_idx]
    # print(f"Selected search query by semantic similarity: {best_query_action}")
    return best_query_action



def select_by_majority_voting(actions):
    """
    Apply majority voting to select most common action
    """
    if len(actions) == 1:
        return actions[0]

    # Count occurrences of each unique action string
    action_counts = Counter(actions)
    
    # Get the most frequent action (Counter.most_common returns [(action, count), ...])
    most_common_action = action_counts.most_common(1)[0][0]

    # print(f"Majority voting result: {most_common_action} with {action_counts[most_common_action]} votes")
    return most_common_action

def select_best_reasoning(think_actions):
    """
    Select the reasoning action that is the most semantically representative
    among the candidates, using sentence embeddings and cosine similarity.
    """
    if len(think_actions) == 1:
        return think_actions[0]

    # Step 1: Extract the inside text from each "think[...]" action
    # Example: "think[This is too expensive]" → "This is too expensive"
    contents = [action[6:-1].strip() for action in think_actions]  # remove 'think[' & ']'

    # Step 2: Convert all reasoning strings into embeddings
    # st_model is a SentenceTransformer that maps text → dense vector
    embeddings = st_model.encode(contents, convert_to_tensor=True)

    # Step 3: Compute the centroid (average vector of all embeddings)
    # This represents the "overall meaning" of all reasoning candidates
    centroid = embeddings.mean(dim=0, keepdim=True)

    # Step 4: Compute cosine similarity between each reasoning and the centroid
    # Higher similarity → reasoning is closer to the "central meaning"
    cos_scores = util.cos_sim(embeddings, centroid)[..., 0]

    # SStep 5: Pick the reasoning with the highest similarity to centroid
    best_idx = int(np.argmax(cos_scores.cpu().numpy()))
    best_reasoning = think_actions[best_idx]

    # print(f"Selected reasoning by semantic similarity: {best_reasoning}")
    #Step 6: Return the most representative reasoning action
    return best_reasoning