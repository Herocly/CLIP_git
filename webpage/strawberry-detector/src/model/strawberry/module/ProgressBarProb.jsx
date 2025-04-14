import {React, useEffect, useState} from 'react';
import './progressBarProb.css'; // 引入CSS文件



function ProgressBarProb({ percent = 0}) { //show progressbar
  
  const progressStyle = {
    width: `${percent}%`,
    backgroundColor: `rgba(${(98*percent + 234*(100-percent))/100},
                          ${(106*percent + 137*(100-percent))/100},
                          ${(255*percent + 184*(100-percent))/100}, 
                          175)`
    //(234,137,184)in 0%   (98,106,255)in 100%
  };
 
  return (
    <div className="progress-container">
      <div className="progress-bar" style={progressStyle}></div>
    </div>
  );
}



 

 
export default ProgressBarProb;