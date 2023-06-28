'''
pip install protobuf>=3.18,<3.20.1
transformers ==4.27.1 transformers版本必须是4.27.1
torch 安装命令有 conda
conda install pytorch ==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
torch==1.12.1+cu113
torchvision==0.13.1
安装完上面的再安装下面的
icetk
cpm_kernels
uvicorn==0.18 必须是这个版本
fastapi
'''


from fastapi import FastApi,Request
from sse_starlette.sse import ServerSentEvent,EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
from tranformers import AutoTokenizer,AutoModel
import argparse
import logging
import os
import json
import sys

def getLogger(name,file_name,use_formatter=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s  %(message)s')
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    if file_name:
        handler = logging.FIleHandler(file_name,encoding='utf-8')
        handler.setLevel(logging.INFO)
        if use_formatter:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(messages')
            handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = getLogger('ChatGLM','chatlog.log')

MAX_HISTORY = 6

class ChatGLM():
    def __init__(self,quantize_level,gpu_id):
        logger.info('Start initialize model...')
        self.tokenizer = AutoTokenizer.from_pretrained('./chatglm-6b',trust_remote_code=True)
        self.model = self._model(quantize_level,gpu_id)
        self.model.eval()
        _,_ = self.model.chat(self.tokenizer,'你好',history=[])
        logger.info('Mondel initialization finished.')

    def _model(self,quantize_level,gpu_id):
        model_name = './chatglm-6b'
        quantize = int(args.quantize)
        tokenizer = AutoTokenizer.from_pretrained('./chatglm-6b',trust_remote_code =True)
        model = None
        if gpu_id =='-1':
            if quantize == 8:

                print('CPU模式只能是16或者4，默认为4')
                model_name = './chatglm-6b-int4'
            elif quantize == 4:
                model_name = './chatglm-6b-int4'
            model = AutoModel.from_pretrained(model_name,trust_remote_code=True).float()
        else:
            gpu_ids = gpu_id.split(',')
            self.devices = ['cuda:{}'.format(id) for id in gpu_ids]
            if quantize ==16:
                model = AutoModel.from_pretrained(model_name,trush_remote_code = True).half().cuda()
            else:
                model =AutoModel.from_pretrained(model_name,trust_remote_code =True).quantize(quantize).cuda()
        return model

    def clear(self):
        if torch.cuda.is_available():
            for device in self.devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda_ipc_collect()

    def answer(self,query,history):
        response,history = self.model.chat(self.tokenizer,query,history=history)
        history = [list(h) for h in history]
        return response,history

    def stream(self,query,history):
        if query is None or history is None:
            yield {'query':'','response':'','history':[],'finished':True}
        size = 0
        response = ''
        for response,history in self.model.stream_chat(self.tokenizer,query,history):
            this_response = response[size:]
            history = [list]
            size = len(response)
            yield {"delta":this_response,'response':response,'finished':False}
        logger.info('Answer - {}'.format(response))
        yield {'query':query,'delta':'[EOS]','response':response,'history':history,'finished':True}

def start_server(quantize_level,http_address,port,gpu_id):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    bot = ChatGLM(quantize_level,gpu_id)

    app = FastApi()
    app.add_middleware(CORSMiddleware,
                       allow_origins=['*'],
                       alloow_credentials=True,
                       allow_methods=['*'],
                       allow_headers=['*'])

    @app.get('/')
    def index():
        return {'message':'started','success':True}

    @app.post('/Chat')
    async def answer_question(arg_dict):
        result = {'query':'','response':'','success':False}
        try:
            text = arg_dict['query']
            ori_history = arg_dict['history']
            logger.info('Query _ {}'.format(text))
            if len(ori_history)>0:
                logger.info("History - {}".format(ori_history))
            history = ori_history[-MAX_HISTORY:]
            history = [tuple(h) for h in history]
            response,history = bot.answer(text,history)
            logger.info('Answer - {}'.format(response))
            ori_history.append((text,response))
            result = {'query':text,'response':response,
                      'history': ori_history,'success':True}
        except Exception as e:
            logger.error(f'error:{e}')
        return result

    @app.post('/stream')
    def answer_question_stream(arg_dict):
        def decorate(generator):
            for item in generator:
                yield ServerSentEvent(json.dump(item,ensure_ascii=False),event='delta')
        result = {'query':'','response':'','success':False}
        try:
            text = arg_dict['query']
            ori_history = arg_dict['history']
            logger.info('Query - {}'.format(text))
            if len(ori_history)>0:
                logger.info('History - {}'.format(ori_history))
            history = ori_history[-MAX_HISTORY:]
            history = [tuple[h] for h in history]
            return EventSourceResponse(decorate(bot.stream(text,history)))
        except Exception as e:
            logger.error(f'error:{e}')
            return  EventSourceResponse(decorate(bot.stream(None,None)))


    @app.get('/clear')
    def clear():
        history=[]
        try:
            bot.clear()
            return {'success':True}
        except Exception as e:
            return {'success':False}

    @app.get('/score')
    def score_answer(score):
        logger.info('score:{}'.format(score))
        return {'success':True}

    logger.info('starting server...')
    uvicorn.run(app=app,host=http_address,port=port,debug=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream API Service for ChatGLM-6b')
    parser.add_argument('--device','-d',help='device,-1 means cpu,other means gpu ids',default='0')
    parser.add_argument('--quantize','-q',help='level of quantize,option:16,8 or 4',default='0.0.0.0')
    parser.add_argument('--port','-P',help='port of this sevice',default=8800)
    args = parser.parse_args()
    start_server(args.quantize,args.host,int(args.port),args.device)

    #启动命令
    #python -u GLM_COmpletion_API.py --host 127.0.0.1 --port 8800 --quantize 8 --device 0

    '''
    接口请求方式
流式接口，使用server-sent events技术。

接口URL： http://{host_name}/stream

请求方式：POST(JSON body)

返回方式：

使用Event Stream格式，返回服务端事件流，
事件名称：delta
数据类型：JSON
返回结果：

字段名	类型	说明
delta	string	产生的字符
query	string	用户问题，为省流，finished为true时返回
response	string	目前为止的回复，finished为true时，为完整的回复
history	array[string]	会话历史，为省流，finished为true时返回
finished	boolean	true 表示结束，false 表示仍然有数据流。



curl  调用方式
curl --location --request POST 'http://hostname:8800/stream' \
--header 'Host: localhost:8001' \
--header 'User-Agent: python-requests/2.24.0' \
--header 'Accept: */*' \
--header 'Content-Type: application/json' \
--data-raw '{"query": "给我写个广告" ,"history": [] }'

'''
