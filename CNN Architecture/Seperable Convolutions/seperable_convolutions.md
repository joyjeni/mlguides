depthwise=torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
 padding=padding,<span color="blue"> groups=in_channels</span>)
 
 
pointwise= torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)


Number of filters used in group = out_channels/in_channels
