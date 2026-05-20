import { Routes, Route, Navigate } from 'react-router-dom'
import DatasetsPage from './pages/DatasetsPage'
import TrainingPage from './pages/TrainingPage'

function App() {
  return (
    <div className="h-screen w-screen overflow-hidden" style={{ minWidth: '100vw' }}>
      <Routes>
        <Route path="/" element={<Navigate to="/datasets" replace />} />
        <Route path="/datasets" element={<DatasetsPage />} />
        <Route path="/training" element={<TrainingPage />} />
      </Routes>
    </div>
  )
}

export default App