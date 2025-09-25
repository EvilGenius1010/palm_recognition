
def segment_hand():

    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="efficientnet-b0",     # Backbone
        encoder_weights="imagenet",         # Pretrained weights
        in_channels=3,                      # RGB input
        classes=1                           # Binary mask (hand vs background)
    )
