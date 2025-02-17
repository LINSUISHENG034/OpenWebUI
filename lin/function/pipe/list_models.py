from pipe_function.deepbricks_pipe_function import Pipe

async def main():
    pipe = Pipe()
    models = pipe.pipes()
    print("Supported models:")
    for model in models:
        print(f"- {model['name']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
