from fastai.vision.all import *
from fastai.imports import *
from fastai.torch_core import *
from fastai.learner import *
from torchvision.models import vit_b_16, ViT_B_16_Weights
@patch
@delegates(subplots)
def plot_metrics(self: Recorder, nrows=None, ncols=None, figsize=None, **kwargs):
    metrics = np.stack(self.values)
    names = self.metric_names[1:-1]
    n = len(names) - 1
    if nrows is None and ncols is None:
        nrows = int(math.sqrt(n))
        ncols = int(np.ceil(n / nrows))
    elif nrows is None: nrows = int(np.ceil(n / ncols))
    elif ncols is None: ncols = int(np.ceil(n / nrows))
    figsize = figsize or (ncols * 6, nrows * 4)
    fig, axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
    axs = [ax if i < n else ax.set_axis_off() for i, ax in enumerate(axs.flatten())][:n]
    for i, (name, ax) in enumerate(zip(names, [axs[0]] + axs)):
        ax.plot(metrics[:, i], color='#1f77b4' if i == 0 else '#ff7f0e', label='valid' if i > 0 else 'train')
        ax.set_title(name if i > 1 else 'losses')
        ax.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")
    # Define paths
    train_path = Path('./training/data/train')
    train_files = get_image_files(train_path)
    # preprocess = weight.transforms()
    # Create a DataLoaders
    dls = ImageDataLoaders.from_folder(train_path,
                                       get_y=parent_label,
                                       valid_pct=0.2,
                                       seed=50,  
                                       item_tfms=Resize(256),                                     
                                       batch_tfms=[*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)],
                                       num_workers = 0,
                                       bs = 128
                                        )
    # print(dls.vocab)
    # model = vgg19_bn(pretrained = True)
    model = resnet50()
    print(model)

    # learn.show_training_loop()
    # print("Accuarcy: " , final_metrics.value[accuracy])
    # print("Precision: " , final_metrics.value[1])
    # print("Recall: " , final_metrics.value[2])
    # print("F1Score: " , final_metrics.value[3])
    # model = swin_v2_b(weights = Swin_V2_B_Weights.IMAGENET1K_V1)
    # learn = Learner(dls, model , metrics=accuracy)

    # learn = vision_learner(dls, densenet121 , metrics=accuracy, pretrained=True, weights= DenseNet121_Weights.IMAGENET1K_V1)
    learn = vision_learner(dls, resnet50 , metrics=accuracy, ps = 0.5, pretrained=True, weights= ResNet50_Weights.IMAGENET1K_V1)
    # learn = vision_learner(dls, resnet50 , metrics=accuracy, pretrained=True, weights= ResNet50_Weights.IMAGENET1K_V2)
    # learn = vision_learner(dls, resnext50_32x4d , metrics=accuracy, pretrained=True, weights= ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
    
    # Fine-tune the model
    # learn.fit_one_cycle(2,slice(1e-3,2e-3), wd = 0.3, cbs=[ShowGraphCallback(), EarlyStoppingCallback(patience=5)])
    learn.fine_tune(epochs=5, wd = 0.3, freeze_epochs = 5, cbs=EarlyStoppingCallback(patience=5))
    # learn.fine_tune(epochs=15)
    learn.recorder.plot_metrics()

    plt.plot(learn.recorder.lrs, learn.recorder.losses)
    plt.title("Learning rate vs Losses")
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.show()

    interp = ClassificationInterpretation.from_learner(learn)
    full_confusion = interp.confusion_matrix()
    # get the most confused matrix
    most_confused_items = interp.most_confused()
    #set the number of most confused type I want to see
    most_confused_items = most_confused_items[:20]
    for item in most_confused_items:
        actual, predicted, occurrences = item
        # get the value in confusion matrix
        actual_index, predicted_index = interp.vocab.o2i[actual], interp.vocab.o2i[predicted]
        confusion_value = full_confusion[actual_index, predicted_index]
        print(f"Actual: {actual}, Predicted: {predicted}, Occurrences: {occurrences}, Confusion Value: {confusion_value}")
        
    current_dir = Path.cwd()
    parent_dir = current_dir.parent
    learn.export(parent_dir/'final_project/training/model/model.pkl')
