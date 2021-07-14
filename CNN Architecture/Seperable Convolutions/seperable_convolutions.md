depthwise=torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
 padding=padding,<span color="blue"> groups=in_channels</span>)
 
 
pointwise= torch.nn.Conv2d(in_channels, out_channels, <span color="blue">kernel_size=1</span>)
