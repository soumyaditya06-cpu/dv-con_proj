import gradio as gr
from pipeline import run_pipeline


TASKS = [
    "step on something",
    "sit comfortably",
    "place flowers",
    "get potatoes out of fire",
    "water plant",
    "get lemon out of tea",
    "dig hole",
    "open bottle of beer",
    "open parcel",
    "serve wine",
    "pour sugar",
    "smear butter",
    "extinguish fire",
    "pound carpet"
]


def app_pipeline(input_image, task):
    temp_path = "/content/temp_input.jpg"
    input_image.save(temp_path)

    _, _, result_img = run_pipeline(temp_path, task)

    return result_img


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Task-Guided Object Selector"
) as demo:

    gr.Markdown("""
    # Task-Guided Object Selection
    Upload an image and select a predefined task.

    The system detects objects using YOLO and selects the most suitable object using MobileCLIP.
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil",
                label="Upload Image"
            )

            task_dropdown = gr.Dropdown(
                choices=TASKS,
                label="Select Task",
                value="place flowers"
            )

            run_btn = gr.Button("Run Detection")

        with gr.Column():
            output_image = gr.Image(
                type="pil",
                label="Selected Object"
            )

    run_btn.click(
        fn=app_pipeline,
        inputs=[input_image, task_dropdown],
        outputs=output_image
    )


if __name__ == "__main__":
    demo.launch(share=True)
