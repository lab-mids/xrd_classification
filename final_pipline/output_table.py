def format_probability(x):
    if x is None:
        return "-"
    if x < 1e-3:
        return f"{x:.2e}"   
    return f"{x:.5f}"
def print_representative_output(
    res,
    cs_class_names,
    sg_class_names,
    true_cs=None,
    true_sg=None,
):
    cs = res["cs"]
    sg = res["sg"]
    comp = res["composition"]

    print("\n" + "="*70)
    print("Representative inference output")
    print("="*70)

    if true_cs is not None or true_sg is not None:
        print(f"True: CS = {true_cs}, SG = {true_sg}")
        print("-"*70)

    # ==================================================
    # TOP BLOCK
    # ==================================================
    print("\nCrystal System (CS)")
    print("-"*40)
    print(f"Final:       {cs_class_names[cs['final_pred']]}")
    print(f"Model:       {cs_class_names[cs['model_pred']]} "
          f"({format_probability(cs['model_confidence'])})")
    print(f"Retrieval:   {cs_class_names[cs['retrieval_pred']]} "
          f"({format_probability(cs['retrieval_confidence'])})")

    if cs["ig_pred"] is not None:
        print(f"IG:          {cs_class_names[cs['ig_pred']]} "
              f"({format_probability(cs['ig_similarity'])})")

    if comp["cs_pred"] is not None:
        print(f"Composition: {cs_class_names[comp['cs_pred']]}")

    print("\nSpace Group (SG)")
    print("-"*40)
    print(f"Final:       {sg_class_names[sg['final_pred']]}")
    print(f"Model:       {sg_class_names[sg['model_pred']]} "
          f"({format_probability(sg['model_confidence'])})")
    print(f"Retrieval:   {sg_class_names[sg['retrieval_pred']]} "
          f"({format_probability(sg['retrieval_confidence'])})")

    if sg["ig_pred"] is not None:
        print(f"IG:          {sg_class_names[sg['ig_pred']]} "
              f"({format_probability(sg['ig_similarity'])})")

    if comp["sg_number"] is not None:
        print(f"Composition: {comp['sg_number']}")

    print("\nAdditional signals")
    print("-"*40)
    print(f"Reconstruction error: {cs['reconstruction_error']:.2e}")

    # ==================================================
    # TOP-K BLOCK
    # ==================================================
    print("\nTop-k candidates (probability / similarity)")
    print("="*70)



    # ---------------- CS IG ----------------
    if cs["tops"]["ig_top"]:
        print("\nCrystal System – IG")
        for i, item in enumerate(cs["tops"]["ig_top"][:5], 1):
            print(f"{i:>2}. {cs_class_names[item['label']]:<15} "
                  f"{format_probability(item['prob'])}")

    # ---------------- CS RETRIEVAL ----------------
    if cs["tops"]["retrieval_top"]:
        print("\nCrystal System – Retrieval")
        for i, item in enumerate(cs["tops"]["retrieval_top"][:5], 1):
            print(f"{i:>2}. {cs_class_names[item['label']]:<15} "
                  f"{format_probability(item['prob'])}")

    # ---------------- CS COMPOSITION ----------------
    if comp["tops"]["cs_top"]:
        print("\nCrystal System – Composition")
        for i, item in enumerate(comp["tops"]["cs_top"][:5], 1):
            print(f"{i:>2}. {cs_class_names[item['label']]:<15} "
                  f"{format_probability(item['prob'])}")


    # ---------------- SG IG ----------------
    if sg["tops"]["ig_top"]:
        print("\nSpace Group – IG")
        for i, item in enumerate(sg["tops"]["ig_top"][:5], 1):
            print(f"{i:>2}. SG {sg_class_names[item['label']]:<5} "
                  f"{format_probability(item['prob'])}")

    # ---------------- SG RETRIEVAL ----------------
    if sg["tops"]["retrieval_top"]:
        print("\nSpace Group – Retrieval")
        for i, item in enumerate(sg["tops"]["retrieval_top"][:5], 1):
            print(f"{i:>2}. SG {sg_class_names[item['label']]:<5} "
                  f"{format_probability(item['prob'])}")

    # ---------------- SG COMPOSITION ----------------
    if comp["tops"]["sg_top_decoded"]:
        print("\nSpace Group – Composition")
        for i, item in enumerate(comp["tops"]["sg_top_decoded"][:5], 1):
            print(f"{i:>2}. SG {item['sg']:<5} "
                  f"{format_probability(item['prob'])}")

    print("\n" + "="*70)