import banana_dev as banana
import base64

model_input={
    "data":"n01440764_tench.jpeg",
}
api_key='27b35759-dbbc-4663-b40c-d84dfd5bb0da'
model_key='03e9ef0f-a043-487e-9cd6-469fbadae2d3'


out= banana.run(api_key,model_key,model_input)

print(out)