"""BLiMP paradigm -> primary overlay-category map, for the acquisition-vs-regularisation
dissociation (Experiment B).

The overlay categories are the primary word categories from categorize.py:
  1 REFERENTIAL   nouns, proper nouns, pronouns, adjectives
  2 PREPOSITION   prepositions, particles
  3 PREDICATIVE   verbs and bare auxiliaries
  4 ADV_CLAUSAL   negation, hedges, amplifiers
  5 COORD_CONJ    coordinating conjunctions
  6 SUBORD_CONJ   subordinating conjunctions / complementisers
  7 DETERMINER    articles, demonstratives, quantifiers  (also wh-determiners: which/what/whose)

Each BLiMP paradigm is attributed to the category(ies) whose overlaid words carry
the grammatical contrast in the minimal pair - i.e. the category an overlay would
have to encode for the model to get that paradigm right. The dissociation test:
overlay a SUBSET of categories (ON) and ask whether accuracy gains concentrate on
paradigms attributed to ON categories vs paradigms attributed to OFF categories.

Two caveats, both stated rather than hidden:
  * DROPPED holds paradigms that test structure/movement (islands, raising,
    filler-gap that hinges on syntax not a word class) or that do not attribute
    cleanly to one overlaid category. They are excluded from the clean read.
  * PREPOSITION (2) has NO BLiMP paradigm that isolates it, so the dissociation
    cannot test category 2. Note this before splitting ON/OFF.

These attributions are linguistic judgement calls (author, 2026-09-24), not ground
truth; the multi-category ones {1,3} etc. are agreement paradigms that genuinely
depend on two categories at once. Revise freely.
"""

# paradigm -> tuple of primary overlay-category ids
PARADIGM_CATEGORIES = {
    # --- DETERMINER (7), det-noun agreement also needs REFERENTIAL (1) ---
    "determiner_noun_agreement_1":                    (7, 1),
    "determiner_noun_agreement_2":                    (7, 1),
    "determiner_noun_agreement_irregular_1":          (7, 1),
    "determiner_noun_agreement_irregular_2":          (7, 1),
    "determiner_noun_agreement_with_adjective_1":     (7, 1),
    "determiner_noun_agreement_with_adj_2":           (7, 1),
    "determiner_noun_agreement_with_adj_irregular_1": (7, 1),
    "determiner_noun_agreement_with_adj_irregular_2": (7, 1),
    "existential_there_quantifiers_1":                (7,),
    "existential_there_quantifiers_2":                (7,),
    "superlative_quantifiers_1":                      (7,),
    "superlative_quantifiers_2":                      (7,),
    # wh-determiners (which/what/whose are DETERMINER in categorize.py)
    "wh_questions_object_gap":                        (7,),
    "wh_questions_subject_gap":                       (7,),
    "wh_questions_subject_gap_long_distance":         (7,),
    "left_branch_island_echo_question":               (7,),
    "left_branch_island_simple_question":             (7,),

    # --- REFERENTIAL (1): anaphora, binding, N-bar ellipsis ---
    "anaphor_gender_agreement":  (1,),
    "anaphor_number_agreement":  (1,),
    "principle_A_c_command":     (1,),
    "principle_A_case_1":        (1,),
    "principle_A_case_2":        (1,),
    "principle_A_domain_1":      (1,),
    "principle_A_domain_2":      (1,),
    "principle_A_domain_3":      (1,),
    "principle_A_reconstruction":(1,),
    "ellipsis_n_bar_1":          (1,),
    "ellipsis_n_bar_2":          (1,),

    # --- PREDICATIVE (3): argument structure, verb forms ---
    "transitive":                     (3,),
    "intransitive":                   (3,),
    "causative":                      (3,),
    "inchoative":                     (3,),
    "drop_argument":                  (3,),
    "passive_1":                      (3,),
    "passive_2":                      (3,),
    "irregular_past_participle_verbs":(3,),

    # --- REFERENTIAL + PREDICATIVE (1,3): subject-verb agreement, animacy ---
    "regular_plural_subject_verb_agreement_1":   (1, 3),
    "regular_plural_subject_verb_agreement_2":   (1, 3),
    "irregular_plural_subject_verb_agreement_1": (1, 3),
    "irregular_plural_subject_verb_agreement_2": (1, 3),
    "distractor_agreement_relational_noun":      (1, 3),
    "distractor_agreement_relative_clause":      (1, 3),
    "animate_subject_passive":                   (1, 3),
    "animate_subject_trans":                     (1, 3),

    # --- ADV_CLAUSAL (4): negation / NPI licensing (npi item any -> DET too) ---
    "sentential_negation_npi_licensor_present": (4,),
    "sentential_negation_npi_scope":            (4,),
    "npi_present_1":                            (4, 7),
    "npi_present_2":                            (4, 7),

    # --- COORD_CONJ (5) ---
    "coordinate_structure_constraint_complex_left_branch": (5,),
    "coordinate_structure_constraint_object_extraction":   (5,),

    # --- SUBORD_CONJ (6): complementiser choice, adjunct clause ---
    "wh_vs_that_no_gap":                (6,),
    "wh_vs_that_no_gap_long_distance":  (6,),
    "wh_vs_that_with_gap":              (6,),
    "wh_vs_that_with_gap_long_distance":(6,),
    "adjunct_island":                   (6,),
}

# Structural / movement / ambiguous - excluded from the clean dissociation read.
DROPPED = {
    "complex_NP_island",
    "wh_island",
    "sentential_subject_island",
    "existential_there_subject_raising",
    "existential_there_object_raising",
    "expletive_it_object_raising",
    "tough_vs_raising_1",
    "tough_vs_raising_2",
    "matrix_question_npi_licensor_present",   # question-licensed NPI: structural
    "only_npi_licensor_present",              # "only" focus, not in the negation set
    "only_npi_scope",
    "irregular_past_participle_adjectives",   # adjectival participle: REF/PRED ambiguous
}

# Categories that have at least one clean paradigm (for choosing ON/OFF splits).
# NOTE: 2 (PREPOSITION) is absent - it has no BLiMP readout.
COVERED_CATEGORIES = sorted({c for cats in PARADIGM_CATEGORIES.values() for c in cats})


def paradigms_for_categories(on_ids, mode="any"):
    """Paradigms attributed to the ON category set.

    mode="any": paradigm counts if ANY of its categories is ON (looser).
    mode="all": paradigm counts only if ALL of its categories are ON (stricter,
                cleaner - a {1,3} agreement paradigm is only 'ON' when both 1 and 3
                are overlaid).
    """
    on = set(on_ids)
    out = []
    for para, cats in PARADIGM_CATEGORIES.items():
        cats = set(cats)
        hit = cats & on
        if (mode == "any" and hit) or (mode == "all" and cats <= on):
            out.append(para)
    return out


if __name__ == "__main__":
    from collections import Counter
    print(f"{len(PARADIGM_CATEGORIES)} paradigms mapped, {len(DROPPED)} dropped, "
          f"{len(PARADIGM_CATEGORIES) + len(DROPPED)} total")
    print(f"covered categories: {COVERED_CATEGORIES}  (2=PREPOSITION has no coverage)")
    solo = Counter(cats[0] for cats in PARADIGM_CATEGORIES.values() if len(cats) == 1)
    print("single-category paradigm counts:", dict(sorted(solo.items())))
