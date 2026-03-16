def print_prediction_results(probabilities, targets, idx_to_label):
    """
    Prints a clean summary of the model's prediction vs reality.
    """
    # Get the last item from the batch
    last_probs = probabilities[-1]
    actual_idx = targets[-1].item()
    
    actual_lang = idx_to_label[actual_idx]
    # Find which language the model actually guessed (highest prob)
    predicted_idx = torch.argmax(last_probs).item()
    predicted_lang = idx_to_label[predicted_idx]

    print("\n" + "="*30)
    print("      MODEL PREDICTION")
    print("="*30)
    print(f"ACTUAL:    {actual_lang}")
    print(f"PREDICTED: {predicted_lang}")
    print("-"*30)
    print("CONFIDENCE SCORES:")

    for i, prob in enumerate(last_probs):
        lang_name = idx_to_label[i]
        bar = "█" * int(prob.item() * 20)  # Visual progress bar
        # Add an arrow to the one the model chose
        pointer = "<-- [Selected]" if i == predicted_idx else ""
        
        print(f"{lang_name:10} | {prob.item():6.2%} {bar.ljust(20)} {pointer}")
    
    print("="*30 + "\n")
