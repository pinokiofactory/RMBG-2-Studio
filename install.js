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
        message:"pip install -r requirements.txt",
      }
    },
    {
      method: "fs.link",
      params: {
        venv: "app/env"
      }
    }
  ]
}

