from fastai.vision.all import *
test_path = Path('./training/data/test')

def generate_predictions(test_folder, output_csv='predictions.csv', model_path=test_path.parent.parent/'model/model.pkl'):
    # Load the exported model
    learn = load_learner(model_path)

    # Get image files from the test folder
    test_files = get_image_files(test_folder)

    # Get predictions for the test set
    test_dl = learn.dls.test_dl(test_files)
    predictions, _ = learn.get_preds(dl=test_dl)

    # Create a DataFrame with the predictions
    pred_labels = [learn.dls.vocab[p] for p in torch.argmax(predictions, axis=1)]
    test_ids = [f.name.replace(".jpg", "") for f in test_files]

    df = pd.DataFrame({'id': test_ids, 'label': pred_labels})

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)

if __name__ == '__main__':
    # Specify the path to the test folder
    # Generate predictions and save to CSV
    generate_predictions(test_path)
