import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'
import StrawberryDetectorMain from './model/strawberry/strawberryDetectorMain'
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App/>
  </StrictMode>
)
