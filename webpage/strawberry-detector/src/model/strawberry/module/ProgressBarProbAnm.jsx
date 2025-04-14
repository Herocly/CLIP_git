
import {React, useEffect, useState} from 'react';
import './progressBarProb.css'; // 引入CSS文件



function ProgressBarProbAnm({ percentStart = 0 , percentEnd = 100, drawtime=1/*second*/  }) {//show from one to another with animation

    const [current_percent, setCurrentPercent] = useState(0);
  
  
    useEffect(()=>{
        const progTimer = setInterval(
          ()=>{
            setCurrentPercent(current_percent + (percentEnd - percentStart)/(100*drawtime));
          },10);
        return () => {
          clearInterval(progTimer);
        }
    }
  )
  
    
    const progressStyle = {
      width: `${current_percent}%`,
      backgroundColor: `rgba(${(98*percent + 234*(100-percent))/100},
                            ${(106*percent + 137*(100-percent))/100},
                            ${(255*percent + 184*(100-percent))/100},
                            175)`
      //(234,137,184)/(98,106,255)
    };
   
    return (
      <div className="progress-container">
        <div className="progress-bar" style={progressStyle}></div>
      </div>
    );
  }

  export default ProgressBarProbAnm;