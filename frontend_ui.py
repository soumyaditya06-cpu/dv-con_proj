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


TASK_PROMPTS = {
    "step on something": "an object suitable to step on safely",
    "sit comfortably": "an object suitable for sitting comfortably",
    "place flowers": "an object suitable for holding flowers",
    "get potatoes out of fire": "a tool used to remove hot food from fire",
    "water plant": "an object used to water a plant",
    "get lemon out of tea": "a utensil used to remove lemon from tea",
    "dig hole": "a tool used for digging a hole",
    "open bottle of beer": "a tool used to open a bottle",
    "open parcel": "a tool used to open a parcel",
    "serve wine": "an object used to serve wine",
    "pour sugar": "an object used to pour sugar",
    "smear butter": "a kitchen utensil used to spread butter",
    "extinguish fire": "an object used to extinguish fire",
    "pound carpet": "an object used to beat or clean a carpet"
}


THRESHOLD = 0.35


def app_pipeline(input_image, task):
    if input_image is None:
        return "Please upload an image.", None

    temp_path = "/content/temp_input.jpg"
    input_image.save(temp_path)

    mapped_prompt = TASK_PROMPTS[task]

    best_obj, scores, result_img = run_pipeline(temp_path, mapped_prompt)

    best_score = max(scores).item()

    if best_score < THRESHOLD:
        return "No suitable object found for the selected task.", None

    selected_name = best_obj["class"].title()

    return f"Selected Object: {selected_name}", result_img


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Task-Guided Object Selection"
) as demo:

    gr.Markdown("""
    # Task-Guided Object Selection
    Upload an image and select a predefined task.

    The system detects objects using YOLO and selects the most suitable object using MobileCLIP.
    """)

    input_image = gr.Image(
        type="pil",
        label="Upload Image"
    )

    task_dropdown = gr.Dropdown(
        choices=TASKS,
        label="Select Task",
        value="place flowers"
    )

    with gr.Row():
        run_btn = gr.Button("Find Best Object")
        clear_btn = gr.Button("Clear")

    status_text = gr.Textbox(
        label="Result",
        interactive=False
    )

    output_image = gr.Image(
        type="pil",
        label="Selected Object",
        visible=True
    )

    run_btn.click(
        fn=app_pipeline,
        inputs=[input_image, task_dropdown],
        outputs=[status_text, output_image],
        show_progress="full"
    )

    clear_btn.click(
        fn=lambda: (None, "place flowers", "", None),
        outputs=[input_image, task_dropdown, status_text, output_image]
    )


if __name__ == "__main__":
    demo.launch(share=True)
