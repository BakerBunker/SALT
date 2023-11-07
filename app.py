from pathlib import Path

import gradio as gr
import pandas as pd

from hubconf import salt
from anonymizer import Anonymizer

def get_demo(anon:Anonymizer):

    with gr.Blocks() as demo:
        state=gr.State(value={
            'fake_spk':anon.get_random_speaker()
        })
        with gr.Tab('interpolate'):
            with gr.Row():
                with gr.Column(variant='compact'):
                    input_audio_file=gr.Audio(sources=['upload','microphone'],type='filepath')
                with gr.Column(scale=2):
                    weight_json=gr.DataFrame(value=state.value['fake_spk'],label="Speaker Weights",row_count=(4,'dynamic'),col_count=(2,'fixed'),interactive=True)
                    rand_spk_btn=gr.Button('Random speaker')
                    with gr.Row():
                        spk_dropdown=gr.Dropdown(list(anon.pool.keys()),label='Preview speaker')
                        spk_sample_audio=gr.Audio(label='Speaker example')
                    weight_slider=gr.Slider(value=0.,minimum=-0.5,maximum=1.5,step=0.01,label='Speaker weight')
                    add_spk_btn=gr.Button("Add speaker")
                    with gr.Row():
                        del_spk_btn=gr.Button("Delete last speaker")
                        norm_spk_btn=gr.Button("Normalize weights to 1")
                        clear_spk_btn=gr.Button("Clear speakers")
            generate_btn=gr.Button('Run!',variant='primary')
            output_audio=gr.Audio(label='Output')
        with gr.Tab('Make speaker pack'):
            upload_file=gr.File(label="Upload speaker wavs in a directory",file_count='directory',file_types=['audio'])
            speaker_name=gr.Textbox(label='Speaker Name')
            gen_pack_btn=gr.Button()
            output_pack_file=gr.File(label="Generated pack file")
        
        def rand_spk_btn_func(stat):
            sdict=anon.get_random_speaker()
            stat['fake_spk']=sdict
            return [stat,sdict]
        rand_spk_btn.click(
            rand_spk_btn_func,inputs=[state],outputs=[state,weight_json]
        )
        def clear_spk_btn_func(stat):
            sdict=pd.DataFrame.from_dict({'speaker':[],'weight':[]})
            stat['fake_spk']=sdict
            return [stat,sdict] 
        clear_spk_btn.click(
            clear_spk_btn_func,inputs=[state],outputs=[state,weight_json]
        ) 
        def add_spk_btn_func(stat,dropdown,slider):
            sdict=stat['fake_spk']
            sdict.loc[len(sdict.index)]=[dropdown,slider]
            stat['fake_spk']=sdict
            return stat,sdict
        add_spk_btn.click(
            add_spk_btn_func,inputs=[state,spk_dropdown,weight_slider],outputs=[state,weight_json]
        )
        def del_spk_btn_func(stat):
            sdict=stat['fake_spk']
            sdict=sdict.iloc[:-1]
            stat['fake_spk']=sdict
            return stat,sdict
        del_spk_btn.click(
            del_spk_btn_func,inputs=[state],outputs=[state,weight_json]
        )
        def norm_spk_btn_func(stat):
            sdict=stat['fake_spk']
            s=sum(sdict['weight'])
            sdict['weight']=sdict['weight'].apply(lambda x:x/s)
            stat['fake_spk']=sdict
            return stat,sdict
        norm_spk_btn.click(
            norm_spk_btn_func,inputs=[state],outputs=[state,weight_json]
        )

        def gen_pack_btn_func(spk_name:str,paths,progress=gr.Progress()):
            if len(spk_name) == 0:
                gr.Warning("Name string is empty")
                return ''
            p=anon.make_speaker_pack([pa.name for pa in paths],spk_name,progress=progress.tqdm)
            anon.add_speaker(spk_name,preprocessed_file=p)
            gr.Info(f'Speaker pack saved to {p} and loaded')
            return p,gr.Dropdown(list(anon.pool.keys()),label='Preview speaker')
        gen_pack_btn.click(
            gen_pack_btn_func,inputs=[speaker_name,upload_file],outputs=[output_pack_file,spk_dropdown]
        )

        spk_dropdown.change(
            lambda name:(anon.pool[name][2],anon.pool[name][1][0].cpu().numpy()),inputs=spk_dropdown,outputs=spk_sample_audio
        )
        

        def generate_func(stat,aud_file):
            wav=anon.interpolate(aud_micaud_file,stat['fake_spk'])
            return 16000,wav.cpu().numpy()
        generate_btn.click(
            generate_func, inputs=[state,input_audio_file],outputs=[output_audio]
        )
    
    return demo

if __name__=='__main__':
    anonymizer=salt(base=True)

    for file in Path('assets').glob('*.pack'):
        anonymizer.add_speaker(name=file.stem,preprocessed_file=file)
    
    get_demo(anonymizer).launch()
