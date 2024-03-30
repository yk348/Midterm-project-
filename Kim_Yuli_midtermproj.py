import pandas as pd
import numpy as np
import itertools
import collections
import os
import mlxtend
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from pandas import DataFrame


# list of dataset names
firm_name = ["amazon", "bestbuy", "K-mart", "nike", "target"]
# Prompting the user to select a store from a list
store = input("please select your store: \n1. amazon\n2. bestbuy\n3. K-mart\n4. nike\n5. target\n")
# loading selected dataset from csv
item = pd.read_csv(firm_name[int(store) -1]+" item names.csv")
tran = pd.read_csv(firm_name[int(store) -1]+" transaction.csv")
# Function to calculate support for each item or itemset based on transactions
def calculate_support(transactions, items):
    total_transactions = len(transactions.index)
    items_with_support = items.copy()
# Initialize the "Support" column with zeros
    items_with_support["Support"] = 0.0
    items_with_support["Support"] = items_with_support["Support"].astype(float)
# Loop through each item in the items DataFrame
    for i in items_with_support.index:
        item_list = items_with_support["Item Name"][i].split(",")
        supp_count = 0
        for j in transactions.index:
            transaction_items = map(str.strip, transactions["Transaction"][j].split(","))
            if all(item in transaction_items for item in item_list):
                supp_count += 1
# Calculate support as the ratio of transactions containing the itemset to total transactions
        items_with_support.loc[i, "Support"] = float(supp_count) / total_transactions
# Sort items based on support values in descending order
    sorted_items = items_with_support.sort_values(by='Support', ascending=False).reset_index(drop=True)
    return sorted_items


# Function to perform brute force item support calculation
def brute_force_item_support(tran, item, min_support):
    k = 1
    dist_items = len(list(set(list(item["Item Name"]))))
    item_support = pd.DataFrame(columns=["Item Name", "Support"])
    all_supports = []  # List to store all new support DataFrames
    while k < dist_items:
        sup_item = pd.DataFrame(columns=["Item Name", "Support"])
        sup_item["Item Name"] = [",".join(i) for i in itertools.combinations(list(item["Item Name"]), k)]
        new_support = calculate_support(tran, sup_item)
        if not new_support.empty:  # Check if new_support is not empty
            all_supports.append(new_support)
        k += 1
    # Concatenate all new support DataFrames
    if all_supports:
        item_support = pd.concat(all_supports)
    # Filter item_support DataFrame to keep itemsets with support greater than min_support
    filtered_item_support = item_support[item_support["Support"] > min_support]
    return filtered_item_support
# Function to calculate confidence for association rule A -> AB
def calculate_confidence(A,AB, item_support_df):
    A_support = item_support_df[item_support_df["Item Name"]==A]["Support"][0]
    AB_support = item_support_df[item_support_df["Item Name"]==AB]["Support"][0]
# Calculate and return the confidence
    return A_support/AB_support


# Generating association rules
def generate_association_rules(pruned_items, min_confidence):
    rules_df = pd.DataFrame(columns=["Association Rules", "Support(%)", "Confidence(%)"])

    # Extract item lists from pruned items
    pruned_items["item_list"] = pruned_items["ItemName"].apply(lambda x: x.split(","))

    # Filter items with more than one element
    eligible_items = pruned_items[pruned_items["item_list"].apply(len) > 1]

    for idx, row in eligible_items.iterrows():
        item_list = row["item_list"]
        # Generate all permutations of item lists
        for perm in itertools.permutations(item_list):
            for n in range(1, len(perm)):
                rule_left, rule_right, support, confidence = calculate_confidence(list(perm[:-n]), perm[-n:],
                                                                                  pruned_items)
                association_rule = f"{rule_left} --> {rule_right}"
                rules_df = rules_df.append({"Association Rules": association_rule,
                                            "Support(%)": support * 100,
                                            "Confidence(%)": confidence * 100},
                                           ignore_index=True)

    # Remove duplicate rules, sort by confidence, and filter by minimum confidence
    rules_df.drop_duplicates(inplace=True)
    rules_df = rules_df.sort_values(by="Confidence(%)", ascending=False).reset_index(drop=True)
    rules_df = rules_df[rules_df["Confidence(%)"] >= min_confidence * 100]

    # Save results to CSV file
    rules_df.to_csv("output/Association_Rules.csv", index=False)

    return rules_df
# Function to calculate confidence of an association rule
min_support = input("please write minimum support i.e. 0.35: ")
min_support = float(min_support)
min_confidence = input("please write minimum confidence i.e. 0.35: ")
min_confidence = float(min_confidence)

asso_rule_df = brute_force_item_support(tran, item, min_support)
print(asso_rule_df)


# Run
def run_apriori_algorithm(items_df, transactions_df, min_sup, min_conf):
    # Find frequent itemsets using Apriori
    low_freq_items, high_freq_items = get_frequent_itemsets_apriori(items_df, transactions_df, min_sup)

    # Remove low frequent itemsets from high frequent itemsets
    high_freq_items = prune_low_freq_itemsets(low_freq_items, high_freq_items)

    # Generate association rules from high frequent itemsets
    association_rules = generate_association_rules_apriori(high_freq_items, min_conf)

    return association_rules


