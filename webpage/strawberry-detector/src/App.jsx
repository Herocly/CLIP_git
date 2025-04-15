import { useState } from 'react';
import React from 'react'
import ReactDom from 'react-dom'
import StrawberryDetectorMain from './model/strawberry/strawberryDetectorMain'
//useStae是一个钩子，用来表示函数的状态

function App() {

    const [isLeftBarShow, setChoice] = useState("mainpage");//储存状态

    const handleSetChoice = (e) => {
        setChoice(e)
    }

    return(
        <div style = {{
            width:'90%' ,
            height:'90%' ,
            marginLeft:'5%'
            }}> 
            <div >

            </div>

            <div style={{height:'100%', marginTop:'5%'}}>
                <StrawberryDetectorMain/>
            </div>
        </div>
    );
}

export default App;