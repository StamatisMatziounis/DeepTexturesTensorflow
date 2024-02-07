<strong>This is an appendix from my thesis report, meaning that it is pretty abstract. That said, it was released in 2022, but it should run with the latest SciPy, Tf, Keras editions. 
Feel free to raise an issue if that's not the case.</strong>

# **Abstract Explanation**

**The functions are categorized into Helper functions, Method functions and
Examples.**

i. The Method functions are the core functions that abstractly refer to the
methods used and use helper functions to implement them. Such functions
are ruinsNormal(), ruinsMin(), ruinsAVG(), ruinsWeightAVG() and
ruinsWeightAVG2. Their common parameters include the feature set
selected, iteration number that can be either constant, or gives the user the
ability to choose when the training ends if they want additional training,
the base image that is the corrupt image, print interval and save interval for
training information displaying. The different part is the Normal method
uses a texture list instead of a texture image. Base image and texture
list/image should be an absolute or relative path to the according textures.

ii. The Example functions are the functions that use the Method functions
and specify the tex_list/img and base_img parameters to the thesis’ results
(specifically the evaluation function).

iii. The helper functions help align the data to the method’s
prerequisites. I.e. the createLoss() function creates a lexicon that contains
the linked layer-index values of the best image for that specific layer.
Another example is the calculatedWeightedScore() function that
implements the Algorithm required for the ruinsWeightAVG() function.
