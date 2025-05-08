import { useState } from 'react';
import React from 'react'
import ReactDom from 'react-dom'
import StrawberryDetectorMain from './model/strawberry/strawberryDetectorMain'
import './App.css'
//useStae是一个钩子，用来表示函数的状态

function App() {

    const [isLeftBarShow, setChoice] = useState("mainpage");//储存状态

    const [isStartPage, SetStartPage] = useState(true);

    const handleSetChoice = (e) => {
        setChoice(e);
    }

    const handleEnterProgram = () => {
        SetStartPage(false);
    }

    return(
        <div style = {{
            width:'90%' ,
            height:'90%' ,
            marginLeft:'5%'
            }}> 

            {
                isStartPage ?
                    (
                    <div className='strawberryDetectorEnter' style={{height:'90%', marginTop:'5%'}}>
                        <div className='titleContainer'>
                            <h1 className="inPageTitle">草莓病害识别系统</h1>
                        </div>

                        <hr className='divideLineUnderTitle'/>

                        <div className='EnterButtonArea'>
                            <button onClick={handleEnterProgram} 
                                className={'EnterButton'}
                                >
                                {'进入系统'}
                            </button>
                        </div>
                    </div>
                    )
                   :
                    (
                    <div style={{height:'90%', marginTop:'5%'}}>
                        <StrawberryDetectorMain/>
                    </div>
                    )
            }
        </div>
    );
}

export default App;