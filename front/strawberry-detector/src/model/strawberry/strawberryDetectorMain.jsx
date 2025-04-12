import { useState, useRef } from 'react';
import './strawberryDetectorMain.css';

function StrawberryDetectorMain() {
    const [image, setImage] = useState(null); // 存储上传的图片
    const [state, setState] = useState(0); // 存放上传状态
    const [state_text, setStateText] = useState(''); // 用于存放识别状态文本
    const [detect_result, setDetectResult] = useState(''); // 用于存放识别结果
    const [loading, setLoading] = useState(false); // 上传状态
    const inputImageUploadRef = useRef(null);

    // 处理图片上传
    const handleImageUpload = (e) => {
        setImage(e.target.files[0]);
        setState(0);
        setStateText('');
        setDetectResult('');
    };

    // 触发上传图片的 input 框点击事件
    const handleUploadImageButtonClick = () => {
        inputImageUploadRef.current.click();
    };

    // 处理图片识别上传逻辑
    const handleUpload = async () => {
        setState(0);
        setStateText('');
        setDetectResult('');

        if (!image) return;

        setLoading(true);
        setState(1);
        setStateText('图片识别中…');

        const formData = new FormData();
        formData.append('image', image);

        try {
            const response = await fetch('http://localhost:5000/upload', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (data.success) {
                setState(1);
                setStateText('识别成功！');
                setDetectResult(`${data.result}`);
            } else {
                setState(-1);
                setStateText(`识别失败，${data.error}`);
            }
        } catch (error) {
            setState(-1);
            setStateText('识别失败，请稍后再试。错误码：' + error);
            console.error(error);
        } finally {
            setLoading(false);
        }

    };


    return (
        <div className='strawberryDetector'>
            <h1>🍓 草莓病害识别系统</h1>

            <div className='imageUploadAndButtonDiv'>
                <label>
                    <input
                        type="file"
                        accept="image/*"
                        id="inputImageUpload"
                        ref={inputImageUploadRef}
                        onChange={handleImageUpload}
                        style={{ display: "none" }}
                    />

                    <div style={{ marginTop: '20px', display: 'flex', marginLeft: "30px" }}>
                        {
                            image ?
                                (<img src={URL.createObjectURL(image)} alt="草莓图片" className='strawberryImgShow' />)
                                :
                                (<div className='noneImgShow'>
                                    <div className='noneImgShowText'>请选择待识别的图片</div>
                                </div>)
                        }
                    </div>
                </label>

                <div className='buttonArea'>
                    <button
                        onClick={handleUploadImageButtonClick}
                        disabled={loading}
                        id='uploadImgButton'
                        style={{ padding: '10px 20px', marginBottom: '10px' }}
                    >
                        {image ? '重新选择' : '选择图片'}
                    </button>

                    <button
                        onClick={handleUpload}
                        disabled={loading || !image}
                        style={{ padding: '10px 20px', marginBottom: '10px', marginLeft: '30px' }}
                    >
                        {loading ? '识别中...' : '识别病害'}
                    </button>
                </div>
            </div>

            <div className='resultDiv'>
                <div className='uploadStateText'>{state_text}</div>
                <div
                    className='detectResult'
                    style={{ display: state !== 0 ? 'initial' : 'none' }}
                >
                    {detect_result}
                </div>
            </div>
        </div>
    );
}
//     return (
//         <div className='strawberryDetector'>
//             <h1>🍓 草莓病害识别系统</h1>
//
//             <div className='imageUploadAndButtonDiv'>
//                 <label>
//                     <input
//                         type="file"
//                         accept="image/*"
//                         id="inputImageUpload"
//                         ref={inputImageUploadRef}
//                         onChange={handleImageUpload}
//                         style={{display: "none"}}
//                     />
//
//                     <div style={{marginTop: '20px', display: 'flex', marginLeft: "30px"}}>
//                         {
//                             image ?
//                                 (<img src={URL.createObjectURL(image)} alt="草莓图片" className='strawberryImgShow'/>)
//                                 :
//                                 (<div className='noneImgShow'>
//                                     <div className='noneImgShowText'>请选择待识别的图片</div>
//                                 </div>)
//                         }
//                     </div>
//                 </label>
//
//                 <div className='buttonArea'>
//                     <button
//                         onClick={handleUploadImageButtonClick}
//                         disabled={loading}
//                         id='uploadImgButton'
//                         style={{padding: '10px 20px', marginBottom: '10px'}}
//                     >
//                         {image ? '重新选择' : '选择图片'}
//                     </button>
//
//                     <button
//                         onClick={handleUpload}
//                         disabled={loading || !image}
//                         style={{padding: '10px 20px', marginBottom: '10px', marginLeft: '30px'}}
//                     >
//                         {loading ? '识别中...' : '识别病害'}
//                     </button>
//                 </div>
//             </div>
//
//             {/* 修改后的结果展示区域 - 并排显示图片和病害信息 */}
//             <div className='resultDisplayArea' style={{
//                 display: state !== 0 ? 'flex' : 'none',
//                 marginTop: '20px',
//                 gap: '30px',
//                 alignItems: 'flex-start'
//             }}>
//                 {/* 图片展示区域 */}
//                 <div className='imageDisplay' style={{flex: 1}}>
//                     {image && <img
//                         src={URL.createObjectURL(image)}
//                         alt="识别图片"
//                         style={{
//                             maxWidth: '300px',
//                             borderRadius: '8px',
//                             boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
//                         }}
//                     />}
//                 </div>
//
//                 {/* 病害信息区域 */}
//                 <div className='diseaseInfo' style={{flex: 1}}>
//                     <h2 style={{color: '#333', marginBottom: '15px'}}>识别结果</h2>
//                     <div className='uploadStateText' style={{
//                         marginBottom: '15px',
//                         color: state === 1 ? '#4CAF50' : state === 2 ? '#F44336' : '#333'
//                     }}>
//                         {state_text}
//                     </div>
//                     <div className='diseaseDetails' style={{
//                         backgroundColor: '#f9f9f9',
//                         padding: '20px',
//                         borderRadius: '8px',
//                         borderLeft: '4px solid #4CAF50'
//                     }}>
//                         <h3 style={{marginTop: 0, color: '#4CAF50'}}>病害类型</h3>
//                         <p style={{fontSize: '18px', fontWeight: 'bold'}}>{detect_result}</p>
//
//                         <h3 style={{marginTop: '20px', color: '#4CAF50'}}>防治建议</h3>
//                         <p>{"应该喷洒农药"}</p>
//                         {/* 这里可以添加更多病害相关信息 */}
//                     </div>
//                 </div>
//             </div>
//         </div>
//     );
// }



export default StrawberryDetectorMain;
