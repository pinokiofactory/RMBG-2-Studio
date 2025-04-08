module.exports = {
  run: [
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",              
          path: "app",
        }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",               
        path: "app",
        message:"uv pip install -r requirements.txt",
                "uv pip install pydantic==2.10.6",
      }
    },
  ]
}

