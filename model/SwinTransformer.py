from network.SwinTransformer import *
def SwinTransformer_Tiny(weights='D:\\Users\\ma\\dachuang\\model\\swin_tiny_patch4_window7_224.pth'):
    model = SwinTransformer(depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24])
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights,map_location=torch.device('cpu'))['model']))
    return model

if __name__ == '__main__':
    device = torch.device('cpu')
    model = SwinTransformer().to(device)
    model.half()
    model.load_state_dict(update_weight(model.state_dict(), torch.load('swin_tiny_patch4_window7_224.pth',map_location=torch.device('cpu'))['model']))
    print(model)

    """
    inputs = torch.randn((1, 3, 640, 512)).to(device).half()
    res = model(inputs)
    for i in res:
        print(i.size())
    print(model.channel)"""