import { useState , useRef , useEffect } from 'react';
import ProgressBarProb from './module/ProgressBarProb';
//useStae是一个钩子，用来表示函数的状态

import './strawberryDetectorMain.css'

function StrawberryDetectorMain() {
    const [image, setImage] = useState(null);//存储上传的图片
    const [state, setState] = useState(0);//存放上传状态
    const [state_text, setStateText] = useState(''); // 用于存放识别状态文本
    const [detect_result, setDetectResult] = useState(''); // 用于存放识别结果

    const [detect_result_type, setDetectResultType] = useState(''); // 用于存放识别结果:识别出的对象
    const [detect_result_prob, setDetectResultProb] = useState(''); // 用于存放识别结果的置信度
    const [detect_result_other, setDetectResultOther] = useState([]); // 用于存放其他可能的结果


    const [loading, setLoading] = useState(false); // 加载状态
    const inputImageUploadRef = useRef(null);

    //handleImageUpload是一个上传图片的函数
    const handleImageUpload = (e) => {
        setImage(e.target.files[0]);//取第一张图

        setState(0); // 每次换图清空识别结果
        setStateText('');
        setDetectResult('');
    };

    const handleUploadImageButtonClick = () =>{
        inputImageUploadRef.current.click() //点击“选择图片”按钮时同样触发上传图片
    }


    const analyzeResultJson = (result) =>{          //解析来自后端的json文本
        setDetectResultType(`${result.predict}`)    //推测的结果
        setDetectResultProb(`${result.prob}`)       //准确度
        setDetectResultOther(result.all)            //所有可能的结果
    }

    //handleDetect包含异步操作，通过async/await来处理异步请求
    const handleStartDetect = async () => {
        setState(0); // 每次换图清空识别结果
        setStateText('');
        setDetectResult('');
        if (!image) return;//检查有无图片
        setLoading(true);
        setState(1);
        setStateText('图片识别中…');

        // 把识别结果都储存在formData中
        const formData = new FormData();
        formData.append('image', image);

        try {
            //response是从后端模拟请求得到的，localhost:5000/detect是假设的后端地址，不必理会
            const response = await fetch('http://localhost:5000/upload', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();


            if(data.success){
                setState(1)
                setStateText('识别成功！');
                analyzeResultJson(data.result)  //识别成功，设置状态为1（fine.）,解析json
            }else{
                setState(-1)
                setStateText(`识别失败，${data.error} `);//识别失败，设置状态为-1，读取错误信息
            }
            
        } catch (error) {
            setStateText('识别失败，请稍后再试。错误码：' + error);
            setState(-1);
            console.error(error);   //识别过程中error处理
        } finally {
            setLoading(false);      //结束加载过程
        } 
    
        // handleDetect();
    };


    // const handleDetect = async() =>{ 
    //     setDetecting(true);
    //     setDetectState(1);
    //     setDetectStateText('识别中…');

    //     const data_post = {option:'detect'}
    
    //     try {
    //         //response是从后端模拟请求得到的，localhost:5000/detect是假设的后端地址，不必理会
    //         const response_detect = await fetch('http://localhost:5000/result', {
    //             method: 'GET'});
    
    //         const data_detect = await response_detect.json();
    
    //         if(data_detect.success){
    //             setDetectState(1);
    //             setDetectStateText('识别成功！');
    //             setDetectResult(`${data_detect.result} `);//识别结果
    //         }else{
    //             setDetectState(-1);
    //             setDetectStateText('识别失败');
    //             setDetectResult(`${data_detect.error} `);//识别结果
    //         }
                
    //     } catch (error) {
    //         setDetectResult('识别出现错误，请稍后再试。错误码：\n' + error);
    //         setDetectState(-1);
    //         console.error(error);
    //     } finally {
    //         setDetecting(false);
    //     }
    // }

    return (
        <div className='strawberryDetector'>
            <h1>🍓 草莓病害识别系统</h1>


            <div className='imageUploadAndButtonDiv'>

                <label>{/*使用label隐藏图片上传input框，用点击图片区域代替*/}

                    <input type="file" accept="image/*"  id="inputImageUpload" ref={inputImageUploadRef} 
                        onChange={handleImageUpload} style={{display:"none"}} />

                    <div style={{ marginTop: '20px', display: 'flex' }}>
                    {
                        image ? 
                        (   <img src={URL.createObjectURL(image)}
                            alt="草莓图片" className='strawberryImgShow'
                            />) //已经上传图片时，展示图片
                        :
                        (   <div className='noneImgShow' >
                                <div className='noneImgShowText'>请选择待识别的图片</div>
                            </div>) //未上传图片时，展示待选择框
                    }
                    </div>

                </label> 

                <div className='buttonArea'>

                    <button onClick={handleUploadImageButtonClick} disabled={loading} 
                        className={`buttonAreaButton ${(loading)?'disabledButton':'enabledButton'/*仅在加载时，选择图片失效*/}}`}
                        style={{ padding: '10px 20px', marginBottom: '10px' }}>
                        {image ?  '重新选择' : '选择图片' }
                    </button>
                    

                    <button onClick={handleStartDetect} disabled={loading || !image} 
                        className={`buttonAreaButton ${(loading || !image)?'disabledButton':'enabledButton'/*加载或无图片时，识别按钮失效*/}}`} 
                        style={{ padding: '10px 20px', marginBottom: '10px'}}>
                        {loading ? '识别中...' : '识别病害'}
                    </button>

                    
                </div>
            </div>

            <div className='resultDiv'>
                <div className={`uploadStateText ${(state!=0)?'--visible':'--hidden'/*状态信息*/}`}
                    style={{marginTop:'10px', fontSize:'20px'}}>{state_text}</div>
                
                <div className={`detectResultType ${(state==1 && !loading)?'--visible':'--hidden'/*识别的结果*/}`} 
                    style={{marginTop:'10px', fontSize:'20px'}}>
                    识别结果：{detect_result_type}
                </div>

                <div className={`detectProb ${(state==1 && !loading)?'--visible':'--hidden'/*最高的置信度*/}`}
                    style={{marginTop:'10px', fontSize:'20px'}}>
                    置信度：{detect_result_prob}%
                </div>
                
                <div className={`allDetectProb ${(state==1 && !loading)?'--visible':'--hidden'/*所有可能性高于0.005%的结果*/}`} 
                    style={{marginTop:'10px', fontSize:'20px'}}>
                        所有可能结果：
                        <table>{/*使用表格格式化地展示所有可能结果*/detect_result_other.map((type) => (
                            <tr key={type.index} 
                                style={{transition: 'opacity .5s ease-in-out',
                                        transitionDelay: `${type.index * 50 + 500/*逐条显示结果*/}ms`}}
                                className={`${(state==1 && !loading)?'--visible':'--hidden'}`}>
                                <td style={{width:'fit-content', float:'left', textAlign:'right'}}>{type.text}:</td>
                                <td style={{marginLeft:'15px', float:'left', minWidth:'70px' ,textAlign:'right'}}>{type.prob}%</td>
                                <td style={{paddingLeft:'15px', float:'left', width:'150px', height:"fit-content"}}>
                                    <ProgressBarProb percent={type.prob}/>
                                    {/* 使用自定义的百分比条可视化比例*/}
                                </td>
                            </tr>
                            ))}
                        </table>
                </div>
            
                

            </div>
        </div>
    );
}

export default StrawberryDetectorMain;