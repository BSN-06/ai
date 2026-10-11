import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import SatelliteAnalysis from './pages/SatelliteAnalysis'
import ThermalAnalysis from './pages/ThermalAnalysis'
import RadarAnalysis from './pages/RadarAnalysis'
import SensorFusion from './pages/SensorFusion'
import ModelManagement from './pages/ModelManagement'
import Training from './pages/Training'
import Evaluation from './pages/Evaluation'
import Datasets from './pages/Datasets'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/"           element={<Dashboard />} />
          <Route path="/satellite"  element={<SatelliteAnalysis />} />
          <Route path="/thermal"    element={<ThermalAnalysis />} />
          <Route path="/radar"      element={<RadarAnalysis />} />
          <Route path="/fusion"     element={<SensorFusion />} />
          <Route path="/models"     element={<ModelManagement />} />
          <Route path="/training"   element={<Training />} />
          <Route path="/evaluation" element={<Evaluation />} />
          <Route path="/datasets"   element={<Datasets />} />
          {/* Fallback */}
          <Route path="*"           element={<Dashboard />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
