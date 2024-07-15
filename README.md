<h1>AI RAG App</h1>h1>

# Steps:

<h3>Install dependencies</h3>
   <p> 1. Do the following before installing the dependencies found in requirements.txt file because of current challenges installing onnxruntime through pip install onnxruntime.</p>
      For Windows users, follow the guide here to install the Microsoft C++ Build Tools. Be sure to follow through to the last step to set the enviroment variable path.
      For MacOS users, a workaround is to first install onnxruntime dependency for chromadb using:
       conda install onnxruntime -c conda-forge
     See this thread for additonal help if needed.
