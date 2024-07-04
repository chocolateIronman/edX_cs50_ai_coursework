import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    probability = 1
    for name in people:
        person = people[name]
        if person['mother'] is not None:
            if person['mother'] in one_gene:
                if person['father'] in one_gene:
                    if person['name'] in one_gene:
                        probability = probability * parent_to_kid(1, 1, 1)
                    elif person['name'] in two_genes:
                        probability = probability * parent_to_kid(1, 1, 2)
                    else:
                        probability = probability * parent_to_kid(1, 1, 0)
                elif person['father'] in two_genes:
                    if person['name'] in one_gene:
                        probability = probability * parent_to_kid(1, 2, 1)
                    elif person['name'] in two_genes:
                        probability = probability * parent_to_kid(1, 2, 2)
                    else:
                        probability = probability * parent_to_kid(1, 2, 0)
                else:
                    if person['name'] in one_gene:
                        probability = probability * parent_to_kid(1, 0, 1)
                    elif person['name'] in two_genes:
                        probability = probability * parent_to_kid(1, 0, 2)
                    else:
                        probability = probability * parent_to_kid(1, 0, 0)
            elif person['mother'] in two_genes:
                if person['father'] in one_gene:
                    if person['name'] in one_gene:
                        probability = probability * parent_to_kid(2, 1, 1)
                    elif person['name'] in two_genes:
                        probability = probability * parent_to_kid(2, 1, 2)
                    else:
                        probability = probability * parent_to_kid(2, 1, 0)
                elif person['father'] in two_genes:
                    if person['name'] in one_gene:
                        probability = probability * parent_to_kid(2, 2, 1)
                    elif person['name'] in two_genes:
                        probability = probability * parent_to_kid(2, 2, 2)
                    else:
                        probability = probability * parent_to_kid(2, 2, 0)
                else:
                    if person['name'] in one_gene:
                        probability = probability * parent_to_kid(2, 0, 1)
                    elif person['name'] in two_genes:
                        probability = probability * parent_to_kid(2, 0, 2)
                    else:
                        probability = probability * parent_to_kid(2, 0, 0)
            else:
                if person['father'] in one_gene:
                    if person['name'] in one_gene:
                        probability = probability * parent_to_kid(0, 1, 1)
                    elif person['name'] in two_genes:
                        probability = probability * parent_to_kid(0, 1, 2)
                    else:
                        probability = probability * parent_to_kid(0, 1, 0)
                elif person['father'] in two_genes:
                    if person['name'] in one_gene:
                        probability = probability * parent_to_kid(0, 2, 1)
                    elif person['name'] in two_genes:
                        probability = probability * parent_to_kid(0, 2, 2)
                    else:
                        probability = probability * parent_to_kid(0, 2, 0)
                else:
                    if person['name'] in one_gene:
                        probability = probability * parent_to_kid(0, 0, 1)
                    elif person['name'] in two_genes:
                        probability = probability * parent_to_kid(0, 0, 2)
                    else:
                        probability = probability * parent_to_kid(0, 0, 0)
            if person['name'] in one_gene:
                if person['name'] in have_trait:
                    probability = probability * PROBS['trait'][1][True]
                else:
                    probability = probability * PROBS['trait'][1][False]
            if person['name'] in two_genes:
                if person['name'] in have_trait:
                    probability = probability * PROBS['trait'][2][True]
                else:
                    probability = probability * PROBS['trait'][2][False]
            if person['name'] not in one_gene and person['name'] not in two_genes:
                if person['name'] in have_trait:
                    probability = probability * PROBS['trait'][0][True]
                else:
                    probability = probability * PROBS['trait'][0][False]
        else:
            if person['name'] in one_gene:
                probability = probability * PROBS['gene'][1]
                if person['name'] in have_trait:
                    probability = probability * PROBS['trait'][1][True]
                else:
                    probability = probability * PROBS['trait'][1][False]
            if person['name'] in two_genes:
                probability = probability * PROBS['gene'][2]
                if person['name'] in have_trait:
                    probability = probability * PROBS['trait'][2][True]
                else:
                    probability = probability * PROBS['trait'][2][False]
            if person['name'] not in one_gene and person['name'] not in two_genes:
                probability = probability * PROBS['gene'][0]
                if person['name'] in have_trait:
                    probability = probability * PROBS['trait'][0][True]
                else:
                    probability = probability * PROBS['trait'][0][False]
    return probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for name in probabilities:
        if name in one_gene:
            probabilities[name]['gene'][1] += p
        elif name in two_genes:
            probabilities[name]['gene'][2] += p
        else:
            probabilities[name]['gene'][0] += p
        if name in have_trait:
            probabilities[name]['trait'][True] += p
        else:
            probabilities[name]['trait'][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for name in probabilities:
        genesTotal = probabilities[name]['gene'][1] + \
            probabilities[name]['gene'][2] + probabilities[name]['gene'][0]
        probabilities[name]['gene'][1] /= genesTotal
        probabilities[name]['gene'][2] /= genesTotal
        probabilities[name]['gene'][0] /= genesTotal

        traitsTotal = probabilities[name]['trait'][True] + \
            probabilities[name]['trait'][False]
        probabilities[name]['trait'][True] /= traitsTotal
        probabilities[name]['trait'][False] /= traitsTotal


def parent_to_kid(mother_gene, father_gene, kid_gene):
    if mother_gene == 2:
        if father_gene == 0:
            # Mother doesn't pass gene (mutated), dad doesn't pass gene
            if kid_gene == 0:
                return PROBS["mutation"] * (1 - PROBS["mutation"])
            # Mother passes gene, dad doesn't pass any or mother doesnt pass gene (mutated), dad passes gene (mutated)
            elif kid_gene == 1:
                return ((1 - PROBS["mutation"]) * (1 - PROBS["mutation"]) + (PROBS["mutation"] * PROBS["mutation"]))
            # Mother passes gene, dad passes gene (mutated)
            elif kid_gene == 2:
                return (1 - PROBS["mutation"]) * PROBS["mutation"]
        elif father_gene == 1:
            # Mother doesn't pass gene (mutated), dad doesn't pass gene (probability)
            if kid_gene == 0:
                return PROBS["mutation"] * 0.5
            # Mother passes gene, dad doesn't pass gene (probability) or Mother doesn't pass gene (mutated), dad passes gene(probability)
            elif kid_gene == 1:
                return ((1 - PROBS["mutation"]) * 0.5) + (PROBS["mutation"] * 0.5)
            # Mother passes gene, dad passes gene (probability)
            elif kid_gene == 2:
                return (1 - PROBS["mutation"]) * 0.5
        elif father_gene == 2:
            # Mother doesn't pass gene (mutated), dad doesn't pass gene (mutated)
            if kid_gene == 0:
                return PROBS["mutation"] * PROBS["mutation"]
            # Mother passes gene, dad doesn't pass gene (mutated) or Mother doesn't pass gene(mutated), dad passes gene
            elif kid_gene == 1:
                return ((1 - PROBS["mutation"]) * PROBS["mutation"]) + (PROBS["mutation"] * (1 - PROBS["mutation"]))
            # Mother passes gene, dad passes gene
            elif kid_gene == 2:
                return (1 - PROBS["mutation"]) * (1 - PROBS["mutation"])
    elif mother_gene == 1:
        if father_gene == 0:
            # Mother doesn't pass the gene (probability), dad doesn't pass gene
            if kid_gene == 0:
                return 0.5 * (1 - PROBS["mutation"])
            # Mother passes gene (probability), dad doesn't pass gene or Mother doesn't pass gene (probability), dad passes gene (mutated)
            elif kid_gene == 1:
                return (0.5 * (1 - PROBS["mutation"])) + (0.5 * PROBS["mutation"])
            # Mother passes gen (probability), dad passes gene (mutated)
            elif kid_gene == 2:
                return 0.5 * PROBS["mutation"]
        elif father_gene == 1:
            # Mother doesn't pass gene (probability), dad doesn't pass gene (probability)
            if kid_gene == 0:
                return 0.5 * 0.5
            # Mother passes gene (probability), dad doesn't pass gene (probability) or Mother doesn't pass gene (probability), dad passes gene(probability)
            elif kid_gene == 1:
                return (0.5 * 0.5) + (0.5 * 0.5)
            # Mother passes gene (probability), dad passes gene (probability)
            elif kid_gene == 2:
                return 0.5 * 0.5
        elif father_gene == 2:
            # Mother doesn't pass gene (probability), dad doesn't pass gene (mutated)
            if kid_gene == 0:
                return 0.5 * PROBS["mutation"]
            # Mother doesn't pass gene (probability), dad passes gene or Mother passes gene(probability), dad doesn't pass gene (mutated)
            elif kid_gene == 1:
                return (0.5 * (1 - PROBS["mutation"])) + (0.5 * PROBS["mutation"])
            # Mother passes gene (probability), dad passes gene
            elif kid_gene == 2:
                return 0.5 * (1 - PROBS["mutation"])
    elif mother_gene == 0:
        if father_gene == 0:
            # Mother doesn't pass gene, dad doesn't pass gene
            if kid_gene == 0:
                return (1 - PROBS["mutation"]) * (1 - PROBS["mutation"])
            # Mother doesn't pass gene,  dad passes gene (mutated) or Mother passes gene (mutated), dad doesn't pass gene
            elif kid_gene == 1:
                return ((1 - PROBS["mutation"]) * PROBS["mutation"]) + (PROBS["mutation"] * (1 - PROBS["mutation"]))
            # Mother passes gene (mutaed), dad passes gene(mutated)
            elif kid_gene == 2:
                return PROBS["mutation"] * PROBS["mutation"]
        elif father_gene == 1:
            # Mother doesn't pass gene, dad passes gene(probability)
            if kid_gene == 0:
                return (1 - PROBS["mutation"]) * 0.5
            # Mother doesn't pass gene, dad passes gene(probability) or Mother passes gene (mutated), dad doesn't pass gene (probability)
            elif kid_gene == 1:
                return ((1 - PROBS["mutation"]) * 0.5) + (PROBS["mutation"] * 0.5)
            # Mother passes gene (mutated), dad passes gene(probability)
            elif kid_gene == 2:
                return PROBS["mutation"] * 0.5
        elif father_gene == 2:
            # Mother doesn't pass gene, dad doesn't pass gene(mutated)
            if kid_gene == 0:
                return (1 - PROBS["mutation"]) * PROBS["mutation"]
            # Mother doesn't pass gene, dad passes gene or Mother passes gene (mutated), dad doesn't pass gene(mutated)
            elif kid_gene == 1:
                return ((1 - PROBS["mutation"]) * (1 - PROBS["mutation"])) + (PROBS["mutation"] * PROBS["mutation"])
            # Mother passes gene (mutated), dad passes gene
            elif kid_gene == 2:
                return PROBS["mutation"] * (1 - PROBS["mutation"])


if __name__ == "__main__":
    main()
