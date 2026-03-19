import ai_hub
import qai_hub as hub

def run(model, input_shape, device):

    model = model.to("cpu").eval()

    device = hub.Device(device)

    traced_model = ai_hub.get_traced_model(input_shape, model)

    compile_job = ai_hub.run_compile(traced_model, device, input_shape)

    profile_job = ai_hub.run_profile(compile_job, device)
    profile_data = profile_job.download_profile()

    return ai_hub.extract_metrics_from_profile(profile_data)

