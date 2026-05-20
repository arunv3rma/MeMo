import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import Plot from 'plotly.js-dist-min'

function TrainingPage() {
  // Helper functions for localStorage
  const saveToStorage = (key, value) => {
    try {
      localStorage.setItem(`training_${key}`, JSON.stringify(value))
    } catch (error) {
      console.warn('Failed to save to localStorage:', error)
    }
  }

  const loadFromStorage = (key, defaultValue) => {
    try {
      const saved = localStorage.getItem(`training_${key}`)
      return saved ? JSON.parse(saved) : defaultValue
    } catch (error) {
      console.warn('Failed to load from localStorage:', error)
      return defaultValue
    }
  }

  // W&B state - entity and project are now configured on server via env vars
  const [wandbRuns, setWandbRuns] = useState(() => loadFromStorage('wandbRuns', []))
  const [loadingWandb, setLoadingWandb] = useState(false)
  const [wandbError, setWandbError] = useState(null)
  const [wandbPage, setWandbPage] = useState(0)
  const [wandbHasMore, setWandbHasMore] = useState(true)
  const [wandbTotalCount, setWandbTotalCount] = useState(0)
  const [wandbFilters, setWandbFilters] = useState(() => loadFromStorage('wandbFilters', {
    tag: '',
    run_id: ''
  }))
  
  // Dashboard state with persistence
  const [dashboards, setDashboards] = useState(() => loadFromStorage('dashboards', []))
  const [selectedDashboard, setSelectedDashboard] = useState(() => loadFromStorage('selectedDashboard', ''))
  const [selectedRun, setSelectedRun] = useState(() => loadFromStorage('selectedRun', null))
  const [dashboardData, setDashboardData] = useState(null)
  const [loadingDashboard, setLoadingDashboard] = useState(false)
  const [plotData, setPlotData] = useState([])
  const [loadingPlots, setLoadingPlots] = useState(false)
  const [selectedTable, setSelectedTable] = useState(null)
  const [selectedTableData, setSelectedTableData] = useState(null)
  const [loadingTableData, setLoadingTableData] = useState(false)
  const [selectedTableExample, setSelectedTableExample] = useState(null)
  const [tableSlices, setTableSlices] = useState([])
  const [activeSlices, setActiveSlices] = useState(new Set())
  const [loadingSlices, setLoadingSlices] = useState(false)
  const [sliceMetricsOverTime, setSliceMetricsOverTime] = useState([])
  const [loadingSliceMetrics, setLoadingSliceMetrics] = useState(false)
  const [selectedSlicesForPlot, setSelectedSlicesForPlot] = useState(new Set())

  // Save state to localStorage whenever it changes

  useEffect(() => {
    saveToStorage('wandbRuns', wandbRuns)
  }, [wandbRuns])

  useEffect(() => {
    saveToStorage('wandbFilters', wandbFilters)
  }, [wandbFilters])

  useEffect(() => {
    saveToStorage('dashboards', dashboards)
  }, [dashboards])

  useEffect(() => {
    saveToStorage('selectedDashboard', selectedDashboard)
  }, [selectedDashboard])

  useEffect(() => {
    saveToStorage('selectedRun', selectedRun)
  }, [selectedRun])

  // Navigate table examples (similar to dataset navigation)
  const navigateTableExample = (direction) => {
    if (!selectedTableExample || !selectedTableData || selectedTableData.length === 0) return
    
    const filteredData = getFilteredTableData()
    if (!filteredData || filteredData.length === 0) return
    
    const currentIndex = filteredData.findIndex(ex => JSON.stringify(ex) === JSON.stringify(selectedTableExample))
    let newIndex
    
    if (direction === 'next') {
      newIndex = (currentIndex + 1) % filteredData.length
    } else {
      newIndex = (currentIndex - 1 + filteredData.length) % filteredData.length
    }
    
    setSelectedTableExample(filteredData[newIndex])
  }

  // Keyboard navigation for table examples
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (!selectedTableExample) return
      
      if (event.key === 'ArrowRight') {
        event.preventDefault()
        navigateTableExample('next')
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault()
        navigateTableExample('prev')
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedTableExample, selectedTableData])

  // Helper function to check if a value is a conversation (array of messages)
  const isConversation = (value) => {
    try {
      if (!Array.isArray(value) || value.length === 0) {
        return false
      }
      
      const isConv = value.every(item => 
        typeof item === 'object' && 
        item !== null && 
        typeof item.role === 'string' && 
        (typeof item.content === 'string' || typeof item.content === 'number')
      )
      
      // Debug logging for problematic cases
      if (!isConv && Array.isArray(value) && value.length > 0) {
        console.log('isConversation failed for:', value)
      }
      
      return isConv
    } catch (error) {
      console.warn('Error checking if value is conversation:', error, value)
      return false
    }
  }

  // Helper function to render conversation messages
  const renderConversation = (messages, title, bgColor, borderColor, textColor) => {
    if (!Array.isArray(messages) || messages.length === 0) {
      return (
        <div className={`p-4 rounded-lg ${bgColor} border ${borderColor}`}>
          <div className={`font-bold text-sm ${textColor} mb-2`}>{title}</div>
          <div className="text-gray-500 text-sm">No conversation data</div>
        </div>
      )
    }

    return (
      <div className={`p-4 rounded-lg ${bgColor} border ${borderColor}`}>
        <div className={`font-bold text-sm ${textColor} mb-4`}>{title}</div>
        <div className="space-y-3">
          {messages.map((message, index) => {
            // Ensure message is an object with the required properties
            if (!message || typeof message !== 'object') {
              console.warn('Invalid message in conversation:', message)
              return null
            }
            
            return (
              <div key={index} className={`p-3 rounded-lg border ${
                message.role === 'user' 
                  ? 'bg-blue-50 border-blue-200' 
                  : message.role === 'system'
                    ? 'bg-yellow-50 border-yellow-200'
                    : 'bg-green-50 border-green-200'
              }`}>
                <div className={`font-semibold text-xs mb-1 ${
                  message.role === 'user' ? 'text-blue-800' : message.role === 'system' ? 'text-yellow-800' : 'text-green-800'
                }`}>
                  {message.role === 'user' ? 'User' : message.role === 'system' ? 'System' : 'Assistant'}
                </div>
                <div className={`text-sm leading-relaxed ${
                  message.role === 'user' ? 'text-blue-900' : message.role === 'system' ? 'text-yellow-900' : 'text-green-900'
                }`}>
                  <ReactMarkdown>{String(message.content || '')}</ReactMarkdown>
                </div>
              </div>
            )
          }).filter(Boolean)}
        </div>
      </div>
    )
  }

  // Get filtered table data based on active slices
  const getFilteredTableData = () => {
    if (!selectedTableData || activeSlices.size === 0) {
      return selectedTableData
    }
    
    // Combine data from all active slices
    const activeSliceData = []
    tableSlices.forEach(slice => {
      if (activeSlices.has(slice.name)) {
        activeSliceData.push(...slice.data)
      }
    })
    
    // Remove duplicates based on a unique identifier (assuming there's an id field or use index)
    // For now, we'll use JSON.stringify to compare objects (not ideal for performance but works)
    const uniqueData = []
    const seenData = new Set()
    
    activeSliceData.forEach(item => {
      const key = JSON.stringify(item)
      if (!seenData.has(key)) {
        seenData.add(key)
        uniqueData.push(item)
      }
    })
    
    return uniqueData
  }

  // W&B functionality using backend API
  const fetchWandbRuns = async (page = 0, append = false) => {
    setLoadingWandb(true)
    setWandbError(null)
    
    try {
      // Get dashboard filters if a dashboard is selected
      const selectedDashboardData = dashboards.find(d => d.name === selectedDashboard)
      const dashboardFilters = selectedDashboardData ? selectedDashboardData.filters : {}
      
      console.log('Fetching W&B runs with:', { selectedDashboard, dashboardFilters, dashboards: dashboards.length, page })
      
      const response = await fetch('/api/wandb/runs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          filters: wandbFilters,
          dashboard_filters: dashboardFilters,
          page: page,
          per_page: 8
        })
      })
      
      const data = await response.json()
      
      if (data.error) {
        throw new Error(data.error)
      }
      
      if (append) {
        setWandbRuns(prevRuns => [...prevRuns, ...data.runs])
      } else {
        setWandbRuns(data.runs)
        setWandbPage(0)
      }
      
      setWandbHasMore(data.has_more || false)
      setWandbTotalCount(data.total || 0)
      setWandbPage(page)
    } catch (error) {
      console.error('Failed to fetch W&B runs:', error)
      setWandbError(`Failed to fetch runs: ${error.message}`)
    } finally {
      setLoadingWandb(false)
    }
  }

  // Load more runs
  const loadMoreRuns = async () => {
    const nextPage = wandbPage + 1
    await fetchWandbRuns(nextPage, true)
  }

  // Fetch runs when dashboard selection changes OR when dashboards are first loaded
  useEffect(() => {
    if (dashboards.length > 0) {
      console.log('useEffect triggered for W&B runs fetch:', { selectedDashboard, dashboardsLength: dashboards.length })
      fetchWandbRuns()
    }
  }, [selectedDashboard, dashboards])

  // Fetch available dashboards
  const fetchDashboards = async () => {
    try {
      const response = await fetch('/api/dashboards')
      const data = await response.json()
      
      if (data.error) {
        console.error('Failed to fetch dashboards:', data.error)
        return
      }
      
      console.log('Fetched dashboards:', data.dashboards.length, 'current selectedDashboard:', selectedDashboard)
      setDashboards(data.dashboards)
      if (data.dashboards.length > 0 && !selectedDashboard) {
        console.log('Auto-selecting first dashboard:', data.dashboards[0].name)
        setSelectedDashboard(data.dashboards[0].name)
      }
    } catch (error) {
      console.error('Failed to fetch dashboards:', error)
    }
  }

  // Auto-fetch dashboards when component mounts (always refresh)
  useEffect(() => {
    // Always fetch fresh dashboards on mount to get latest specs
    fetchDashboards()
  }, [])

  // Debug log on mount to see restored state
  useEffect(() => {
    console.log('TrainingPage mounted with restored state:', {
      selectedDashboard,
      dashboardsCount: dashboards.length,
      runsCount: wandbRuns.length
    })
  }, [])

  // Auto-fetch runs on mount if we have all the required data persisted
  useEffect(() => {
    if (dashboards.length > 0 && wandbRuns.length === 0 && selectedDashboard) {
      console.log('Auto-fetching runs on mount with persisted data')
      fetchWandbRuns()
    }
  }, [])  // Only run on mount

  // Auto-analyze restored selected run when we have all the data
  useEffect(() => {
    if (selectedRun && selectedDashboard && dashboards.length > 0 && !dashboardData && !loadingDashboard) {
      console.log('Auto-analyzing restored selected run:', selectedRun.id)
      analyzeRun(selectedRun)
    }
  }, [selectedRun, selectedDashboard, dashboards.length, dashboardData, loadingDashboard])

  // Analyze run with selected dashboard
  const analyzeRun = async (run) => {
    if (!selectedDashboard) {
      console.error('No dashboard selected')
      return
    }

    // Clear all cached data FIRST before setting new run
    setDashboardData(null)
    setPlotData([])  // Clear previous plot data
    setSelectedTable(null)
    setSelectedTableData(null)  // Clear previous table data
    setSelectedTableExample(null)
    setSliceMetricsOverTime([])  // Clear previous slice metrics
    setTableSlices([])  // Clear previous table slices
    setActiveSlices(new Set())  // Clear active slices
    setSelectedSlicesForPlot(new Set())  // Clear selected slices for plot
    setLoadingSliceMetrics(false)  // Reset loading state
    
    // Now set the new run and loading state
    setSelectedRun(run)
    setLoadingDashboard(true)
    
    try {
      const response = await fetch('/api/dashboard/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          run_id: run.id,
          dashboard_name: selectedDashboard
        })
      })
      
      const data = await response.json()
      
      if (data.error) {
        throw new Error(data.error)
      }
      
      console.log('Dashboard data loaded:', data)
      setDashboardData(data)
      
      // Start loading plots asynchronously (don't await)
      console.log('Starting async plot loading...')
      loadPlotData(run, selectedDashboard)
      
      // Start loading slice metrics over time in background (don't await)
      console.log('Starting async slice metrics loading...')
      loadSliceMetricsOverTime(run, selectedDashboard)
      
      if (data.tables.length > 0) {
        // Select the last table (highest step) by default since that's usually the final results
        const lastTable = data.tables[data.tables.length - 1]
        console.log('Loading last table by default:', lastTable)
        setSelectedTable(lastTable)
        loadTableData(lastTable, run)  // Load the last table's data, passing run directly
        loadTableSlices(lastTable, run, selectedDashboard)  // Load slices for the table
      } else {
        console.log('No tables found in dashboard data')
      }
    } catch (error) {
      console.error('Failed to analyze run:', error)
      setWandbError(`Failed to analyze run: ${error.message}`)
    } finally {
      setLoadingDashboard(false)
    }
  }

  // Load plot data asynchronously
  const loadPlotData = async (run = selectedRun, dashboard = selectedDashboard) => {
    console.log('loadPlotData called', { run, dashboard })
    if (!run || !dashboard) {
      console.log('Missing run or dashboard, skipping plot loading')
      return
    }

    setLoadingPlots(true)
    console.log('Loading plots for run:', run.id, 'dashboard:', dashboard)
    
    try {
      const response = await fetch('/api/dashboard/plots', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          run_id: run.id,
          dashboard_name: dashboard
        })
      })
      
      const data = await response.json()
      console.log('Plot data response:', data)
      
      if (data.error) {
        throw new Error(data.error)
      }
      
      console.log('Setting plot data:', data.plots)
      setPlotData(data.plots)
    } catch (error) {
      console.error('Failed to load plot data:', error)
      setWandbError(`Failed to load plot data: ${error.message}`)
    } finally {
      setLoadingPlots(false)
    }
  }

  // Load slice metrics over time for all table steps
  const loadSliceMetricsOverTime = async (run = selectedRun, dashboard = selectedDashboard) => {
    console.log('loadSliceMetricsOverTime called with:', { run: run?.id, dashboard })
    if (!run || !dashboard) {
      console.log('Missing run or dashboard, skipping slice metrics loading')
      return
    }

    setLoadingSliceMetrics(true)
    console.log('Loading slice metrics over time for all steps...')
    
    try {
      const response = await fetch('/api/dashboard/slice-metrics', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          run_id: run.id,
          dashboard_name: dashboard
        })
      })
      
      const data = await response.json()
      
      if (data.error) {
        throw new Error(data.error)
      }
      
      console.log('Slice metrics over time loaded successfully:', data.slice_metrics.length, 'slices across', data.step_count, 'steps')
      setSliceMetricsOverTime(data.slice_metrics)
    } catch (error) {
      console.error('Failed to load slice metrics over time:', error)
      setWandbError(`Failed to load slice metrics over time: ${error.message}`)
    } finally {
      setLoadingSliceMetrics(false)
    }
  }

  // Load slices for a table
  const loadTableSlices = async (table, run = selectedRun, dashboard = selectedDashboard) => {
    console.log('loadTableSlices called with:', { run: run?.id, table, dashboard })
    if (!run || !table || !dashboard) {
      console.log('Missing run, table, or dashboard, skipping slice loading')
      return
    }

    setLoadingSlices(true)
    console.log('Loading slices for step:', table.step)
    
    try {
      const response = await fetch('/api/dashboard/slices', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          run_id: run.id,
          dashboard_name: dashboard,
          table_path: table.path,
          table_step: table.step
        })
      })
      
      const data = await response.json()
      
      if (data.error) {
        throw new Error(data.error)
      }
      
      console.log('Slices loaded successfully:', data.slices.length, 'slices')
      setTableSlices(data.slices)
      setActiveSlices(new Set()) // Reset active slices when loading new data
    } catch (error) {
      console.error('Failed to load table slices:', error)
      setWandbError(`Failed to load table slices: ${error.message}`)
    } finally {
      setLoadingSlices(false)
    }
  }

  // Load table data on demand
  const loadTableData = async (table, run = selectedRun) => {
    console.log('loadTableData called with:', { run: run?.id, table })
    if (!run || !table) {
      console.log('Missing run or table, skipping table data loading')
      return
    }

    setLoadingTableData(true)
    console.log('Loading table data for step:', table.step)
    
    try {
      const response = await fetch('/api/dashboard/table', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          run_id: run.id,
          table_path: table.path,
          table_step: table.step
        })
      })
      
      const data = await response.json()
      
      if (data.error) {
        throw new Error(data.error)
      }
      
      console.log('Table data loaded successfully:', data.data.length, 'rows')
      setSelectedTableData(data.data)
    } catch (error) {
      console.error('Failed to load table data:', error)
      setWandbError(`Failed to load table data: ${error.message}`)
    } finally {
      setLoadingTableData(false)
    }
  }

  // Plot visualization component
  const PlotVisualization = ({ plot, sliceMetrics, selectedSlices }) => {
    const plotRef = useRef(null)
    
    useEffect(() => {
      if (!plotRef.current || !plot) return
      
      const traces = []
      
      // Main plot trace
      const mainTrace = {
        x: plot.data.map(d => d[plot.x_col]),
        y: plot.data.map(d => d[plot.y_col]),
        type: 'scatter',
        mode: 'lines+markers',
        name: plot.plot_name,
        line: { color: '#3b82f6', width: 2 },
        marker: { color: '#3b82f6', size: 6 }
      }
      traces.push(mainTrace)
      
      // Add slice traces if available and selected
      if (sliceMetrics && selectedSlices && selectedSlices.size > 0) {
        const colors = ['#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#f97316', '#06b6d4']
        let colorIndex = 0
        
        sliceMetrics.forEach(slice => {
          if (selectedSlices.has(slice.name)) {
            // Find the metric that matches the plot's y-axis
            const metricKey = Object.keys(slice.data[0] || {}).find(key => 
              key !== 'step' && key.includes(plot.y_col.split('/')[1] || plot.y_col)
            )
            
            if (metricKey && slice.data.length > 0) {
              const color = colors[colorIndex % colors.length]
              colorIndex++
              
              const sliceTrace = {
                x: slice.data.map(d => d.step),
                y: slice.data.map(d => d[metricKey]),
                type: 'scatter',
                mode: 'lines+markers',
                name: slice.name,
                line: { color: color, width: 2, dash: 'dot' },
                marker: { color: color, size: 4 }
              }
              traces.push(sliceTrace)
            }
          }
        })
      }
      
      const layout = {
        title: {
          text: plot.plot_name,
          font: { size: 14 }
        },
        xaxis: { 
          title: plot.x_col,
          gridcolor: '#f3f4f6'
        },
        yaxis: { 
          title: plot.y_col,
          gridcolor: '#f3f4f6'
        },
        margin: { l: 50, r: 30, t: 40, b: 40 },
        plot_bgcolor: '#fafafa',
        paper_bgcolor: 'white',
        legend: {
          orientation: 'h',
          x: 0.5,
          xanchor: 'center',
          y: 1.02,
          yanchor: 'bottom'
        }
      }
      
      const config = {
        displayModeBar: false,
        responsive: true
      }
      
      Plot.newPlot(plotRef.current, traces, layout, config)
    }, [plot, sliceMetrics, selectedSlices])
    
    if (!plot) return null
    
    return <div ref={plotRef} className="w-full h-80" />
  }

  return (
    <div className="flex h-full font-sans overflow-hidden" style={{ width: '100vw', minWidth: '100vw' }}>
      <div className="w-96 bg-gray-100 border-r border-gray-300 flex flex-col flex-shrink-0 overflow-hidden">
        {/* Navigation Tabs */}
        <div className="p-4 border-b border-gray-300">
          <div className="flex space-x-1">
            <a
              href="/datasets"
              className="px-3 py-2 text-sm font-medium text-gray-600 hover:text-gray-800 hover:bg-gray-200 rounded transition-colors"
            >
              Datasets
            </a>
            <div className="px-3 py-2 text-sm font-medium text-blue-600 bg-blue-100 rounded">
              Training
            </div>
          </div>
        </div>

        {/* Dashboard Selection */}
        <div className="p-4 border-b border-gray-300">
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Dashboard
              </label>
              <select
                value={selectedDashboard}
                onChange={(e) => {
                  const newDashboard = e.target.value
                  console.log('Dashboard changed:', selectedDashboard, '->', newDashboard)
                  
                  // Clear all cached data including runs (since they may have cached slice scores)
                  setDashboardData(null)
                  setPlotData([])
                  setSelectedTable(null)
                  setSelectedTableData(null)
                  setSelectedTableExample(null)
                  setTableSlices([])
                  setActiveSlices(new Set())
                  setSliceMetricsOverTime([])
                  setSelectedSlicesForPlot(new Set())
                  setWandbRuns([])  // Clear runs since they may have cached slice scores
                  setSelectedRun(null)  // Clear selected run
                  
                  // Update dashboard selection
                  setSelectedDashboard(newDashboard)
                  
                  // Fetch fresh runs with new dashboard filters
                  if (newDashboard) {
                    console.log('Fetching fresh runs for new dashboard:', newDashboard)
                    fetchWandbRuns()
                  }
                }}
                className="w-full p-2 border border-gray-300 rounded text-sm"
              >
                <option value="">Select a dashboard</option>
                {dashboards.map((dashboard) => (
                  <option key={dashboard.name} value={dashboard.name}>
                    {dashboard.name}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* W&B Configuration */}
        <div className="p-4">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Filter by Tag
              </label>
              <input
                type="text"
                value={wandbFilters.tag}
                onChange={(e) => setWandbFilters({...wandbFilters, tag: e.target.value})}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    fetchWandbRuns()
                  }
                }}
                placeholder="e.g., generate, paper"
                className="w-full p-2 border border-gray-300 rounded text-sm"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Filter by Run ID/Name
              </label>
              <input
                type="text"
                value={wandbFilters.run_id}
                onChange={(e) => setWandbFilters({...wandbFilters, run_id: e.target.value})}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    fetchWandbRuns()
                  }
                }}
                placeholder="e.g., longhealth, p10"
                className="w-full p-2 border border-gray-300 rounded text-sm"
              />
            </div>
          </div>
        </div>

        {/* Error Display */}
        {wandbError && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
            <div className="text-sm text-red-800">{wandbError}</div>
          </div>
        )}


        {/* W&B Runs */}
        {(wandbRuns.length > 0 || (!loadingWandb && dashboards.length > 0)) && (
          <div className="flex-1 flex flex-col min-h-0">
            <div className="flex items-center justify-between mb-3 px-4">
              <h3 className="text-sm font-medium text-gray-700">
                Recent Runs ({wandbRuns.length}{wandbTotalCount > 0 ? ` of ${wandbTotalCount}` : ''})
              </h3>
              <div className="flex items-center gap-2">
                {loadingWandb && (
                  <div className="text-xs text-gray-500">Loading...</div>
                )}
                <button
                  onClick={() => fetchWandbRuns()}
                  disabled={loadingWandb}
                  className={`px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 disabled:bg-gray-50 disabled:text-gray-400 text-gray-700 rounded transition-colors ${
                    loadingWandb ? 'animate-spin' : ''
                  }`}
                  title="Refresh runs"
                >
                  ↻
                </button>
              </div>
            </div>
            <div className="space-y-2 flex-1 overflow-y-auto px-4">
              {wandbRuns.length > 0 ? wandbRuns.map((run) => (
                <div
                  key={run.id}
                  className={`p-3 rounded-lg cursor-pointer transition-all border ${
                    selectedRun?.id === run.id ? 'bg-blue-50 border-blue-300' : ''
                  }`}
                  onClick={() => analyzeRun(run)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="font-medium text-sm text-gray-800 truncate">
                      {run.name}
                    </div>
                    <div className={`px-2 py-1 text-xs rounded-full ${
                      run.state === 'running' ? 'bg-blue-100 text-blue-800' :
                      run.state === 'finished' ? 'bg-green-100 text-green-800' :
                      run.state === 'failed' ? 'bg-red-100 text-red-800' :
                      run.state === 'crashed' ? 'bg-red-100 text-red-800 bg-red-200' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {run.state}
                    </div>
                  </div>
                  <button
                    className="text-xs text-gray-500 mb-1 hover:text-gray-700 hover:bg-gray-100 px-2 py-1 rounded transition-colors cursor-pointer text-left"
                    onClick={(e) => {
                      e.stopPropagation()
                      navigator.clipboard.writeText(run.id).then(() => {
                        // Optional: Add visual feedback here
                        const button = e.target
                        const originalText = button.textContent
                        button.textContent = 'Copied!'
                        setTimeout(() => {
                          button.textContent = originalText
                        }, 1000)
                      }).catch(err => {
                        console.error('Failed to copy: ', err)
                      })
                    }}
                    title="Click to copy ID"
                  >
                    ID: {run.id}
                  </button>
                  {run.tags && run.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1 mb-2">
                      {run.tags.slice(0, 3).map((tag, idx) => (
                        <span key={idx} className="px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded">
                          {tag}
                        </span>
                      ))}
                      {run.tags.length > 3 && (
                        <span className="text-xs text-gray-500">+{run.tags.length - 3} more</span>
                      )}
                    </div>
                  )}
                  <div className="text-xs text-gray-400">
                    Created: {new Date(run.createdAt).toLocaleDateString()}
                  </div>
                  <a 
                    href={run.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-xs text-blue-600 hover:text-blue-800 mt-1 inline-block"
                    onClick={(e) => e.stopPropagation()}
                  >
                    View in W&B →
                  </a>
                </div>
              )) : (
                <div className="text-center text-gray-500 py-8">
                  <p className="text-sm">No runs found.</p>
                  <p className="text-xs mt-1">Use filters above and click refresh to fetch runs.</p>
                </div>
              )}
            </div>
            
            {/* Load More Button */}
            {wandbRuns.length > 0 && wandbHasMore && (
              <div className="px-4 pb-4">
                <button
                  onClick={loadMoreRuns}
                  disabled={loadingWandb}
                  className={`w-full px-3 py-2 text-sm rounded border transition-colors ${
                    loadingWandb 
                      ? 'bg-gray-50 border-gray-200 text-gray-400 cursor-not-allowed'
                      : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50 hover:border-gray-400'
                  }`}
                >
                  {loadingWandb ? 'Loading...' : 'Load More'}
                </button>
              </div>
            )}
          </div>
        )}

        {/* W&B API Configuration Note */}
        <div className="p-4 mt-auto">
          <div className="text-xs text-gray-600 bg-blue-50 p-2 rounded">
            W&B API Key, Entity, and Project are configured on the server via environment variables.
          </div>
        </div>
      </div>
      
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden" style={{ width: 'calc(100vw - 24rem)' }}>
        {selectedRun && dashboardData ? (
          <div className="flex flex-col h-full w-full overflow-hidden">

            {/* Plots Section */}
            {(dashboardData?.plots?.length > 0 || plotData.length > 0 || loadingPlots) && (
              <div className="bg-white p-4 border-b border-gray-200 w-full flex-shrink-0">
                {loadingSliceMetrics && (
                  <div className="text-sm text-gray-500 mb-4">Loading slice metrics...</div>
                )}
                
                {/* Slice Selection for Plots */}
                {!loadingSliceMetrics && sliceMetricsOverTime.length > 0 && (
                  <div className="mb-4 p-3 bg-gray-50 border border-gray-200 rounded">
                    <div className="text-sm font-medium text-gray-700 mb-2">
                      Show slice metrics in plot:
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {sliceMetricsOverTime.map((slice, idx) => (
                        <button
                          key={idx}
                          onClick={() => {
                            const newSelected = new Set(selectedSlicesForPlot)
                            if (selectedSlicesForPlot.has(slice.name)) {
                              newSelected.delete(slice.name)
                            } else {
                              newSelected.add(slice.name)
                            }
                            setSelectedSlicesForPlot(newSelected)
                          }}
                          className={`px-3 py-1 text-xs rounded-full border transition-colors ${
                            selectedSlicesForPlot.has(slice.name)
                              ? 'bg-blue-100 border-blue-300 text-blue-800'
                              : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-100'
                          }`}
                        >
                          {slice.name}
                        </button>
                      ))}
                      {selectedSlicesForPlot.size > 0 && (
                        <button
                          onClick={() => setSelectedSlicesForPlot(new Set())}
                          className="px-3 py-1 text-xs rounded-full border border-red-300 text-red-700 bg-red-50 hover:bg-red-100 transition-colors ml-2"
                        >
                          Clear
                        </button>
                      )}
                    </div>
                  </div>
                )}
                {loadingPlots && plotData.length === 0 ? (
                  <div className="flex items-center justify-center p-8 w-full">
                    <div className="text-gray-500">Loading plots...</div>
                  </div>
                ) : (
                  <div className="w-full">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 w-full">
                      {plotData.map((plot, idx) => (
                        <div key={idx} className="border border-gray-200 rounded-lg p-4 w-full min-w-0">
                          <PlotVisualization 
                            plot={plot} 
                            sliceMetrics={sliceMetricsOverTime}
                            selectedSlices={selectedSlicesForPlot}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Tables Section */}
            {dashboardData.tables && dashboardData.tables.length > 0 && (
              <div className="flex-1 flex flex-col min-h-0 w-full overflow-hidden">
                {/* Step Selector and Slices Section */}
                <div className="bg-white p-4 border-b border-gray-200 w-full flex-shrink-0">
                  <div className="flex items-center gap-4 mb-3">
                    <div className="flex items-center gap-2">
                      <label className="text-sm font-medium text-gray-700">Step:</label>
                      <select
                        value={selectedTable ? dashboardData.tables.indexOf(selectedTable) : 0}
                        onChange={(e) => {
                          const newTable = dashboardData.tables[parseInt(e.target.value)]
                          setSelectedTable(newTable)
                          setSelectedTableData(null)  // Clear previous data
                          setSelectedTableExample(null)  // Clear selected example
                          setTableSlices([])  // Clear previous slices
                          setActiveSlices(new Set())  // Clear active slices
                          loadTableData(newTable)  // Load new table data
                          loadTableSlices(newTable)  // Load slices for new table
                        }}
                        className="p-2 border border-gray-300 rounded text-sm"
                      >
                        {dashboardData.tables.map((table, idx) => (
                          <option key={idx} value={idx}>
                            {table.step}
                          </option>
                        ))}
                      </select>
                    </div>
                    {loadingSlices && (
                      <div className="text-sm text-gray-500">Loading slices...</div>
                    )}
                  </div>
                  
                  {/* Slices */}
                  {tableSlices.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {tableSlices.map((slice, idx) => (
                        <div
                          key={idx}
                          className={`border rounded-lg p-3 transition-colors cursor-pointer ${
                            activeSlices.has(slice.name)
                              ? 'bg-blue-50 border-blue-300'
                              : 'bg-gray-50 border-gray-300 hover:bg-gray-100'
                          }`}
                          onClick={() => {
                            const newActiveSlices = new Set(activeSlices)
                            if (activeSlices.has(slice.name)) {
                              newActiveSlices.delete(slice.name)
                            } else {
                              newActiveSlices.add(slice.name)
                            }
                            setActiveSlices(newActiveSlices)
                          }}
                        >
                          <div className={`font-medium text-sm ${
                            activeSlices.has(slice.name) ? 'text-blue-800' : 'text-gray-800'
                          }`}>
                            {slice.name}
                          </div>
                          <div className={`text-xs mt-1 ${
                            activeSlices.has(slice.name) ? 'text-blue-600' : 'text-gray-600'
                          }`}>
                            {slice.count} examples
                          </div>
                          {slice.metrics && Object.keys(slice.metrics).length > 0 && (
                            <div className={`text-xs mt-2 space-y-1 ${
                              activeSlices.has(slice.name) ? 'text-blue-700' : 'text-gray-700'
                            }`}>
                              {Object.entries(slice.metrics).map(([key, value]) => (
                                <div key={key} className="flex justify-between">
                                  <span className="font-medium">{key}:</span>
                                  <span>
                                    {value !== null ? 
                                      (typeof value === 'number' ? value.toFixed(3) : value) : 
                                      'N/A'
                                    }
                                  </span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                      {activeSlices.size > 0 && (
                        <div
                          onClick={() => setActiveSlices(new Set())}
                          className="border rounded-lg p-3 transition-colors cursor-pointer bg-red-50 border-red-300 hover:bg-red-100"
                        >
                          <div className="font-medium text-sm text-red-800">
                            Clear All
                          </div>
                          <div className="text-xs mt-1 text-red-600">
                            Reset filters
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {!selectedTableExample ? (
                  /* Table Gallery View */
                  <div className="flex-1 overflow-y-auto p-4 w-full">
                    {loadingTableData ? (
                      <div className="flex flex-col items-center justify-center py-12 text-center text-gray-600 w-full">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-4"></div>
                        <p>Loading table data...</p>
                      </div>
                    ) : selectedTableData && selectedTableData.length > 0 ? (
                      <div className="w-full">
                        {activeSlices.size > 0 && (
                          <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded">
                            <div className="text-sm text-blue-800">
                              Showing {getFilteredTableData().length} examples from {activeSlices.size} active slice{activeSlices.size !== 1 ? 's' : ''}: {Array.from(activeSlices).join(', ')}
                            </div>
                          </div>
                        )}
                        {getFilteredTableData().length > 0 ? (
                          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 w-full">
                            {getFilteredTableData().map((example, idx) => (
                            <div
                              key={idx}
                              className="border border-gray-300 rounded-lg p-4 cursor-pointer transition-all hover:-translate-y-1 hover:shadow-lg w-full min-h-[200px]"
                              onClick={() => setSelectedTableExample(example)}
                            >
                              <div className="mb-2">
                                {(() => {
                                  const score = example[selectedTable.score_col]
                                  if (score === null || score === undefined || score === '') {
                                    return (
                                      <span className="px-2 py-1 text-xs rounded-full bg-gray-200 text-gray-600">
                                        N/A
                                      </span>
                                    )
                                  }
                                  
                                  // Handle boolean values
                                  if (typeof score === 'boolean' || score === 'true' || score === 'false') {
                                    const isTrue = score === true || score === 'true'
                                    return (
                                      <span className={`px-2 py-1 text-xs rounded-full font-medium text-white ${
                                        isTrue ? 'bg-green-500' : 'bg-red-500'
                                      }`}>
                                        {isTrue ? 'CORRECT' : 'INCORRECT'}
                                      </span>
                                    )
                                  }
                                  
                                  const numScore = parseFloat(score)
                                  if (isNaN(numScore)) {
                                    return (
                                      <span className="px-2 py-1 text-xs rounded-full bg-gray-200 text-gray-600">
                                        N/A
                                      </span>
                                    )
                                  }
                                  
                                  // Create gradient from red (0) to yellow (0.5) to green (1)
                                  let bgColor, textColor
                                  if (numScore < 0.5) {
                                    // Red to yellow gradient
                                    const intensity = numScore * 2 // 0 to 1
                                    bgColor = `rgb(${255}, ${Math.round(165 + (255 - 165) * intensity)}, 0)`
                                    textColor = 'text-white'
                                  } else {
                                    // Yellow to green gradient
                                    const intensity = (numScore - 0.5) * 2 // 0 to 1
                                    bgColor = `rgb(${Math.round(255 - 255 * intensity)}, 255, 0)`
                                    textColor = intensity > 0.3 ? 'text-white' : 'text-black'
                                  }
                                  
                                  return (
                                    <span 
                                      className={`px-2 py-1 text-xs rounded-full font-medium ${textColor}`}
                                      style={{ backgroundColor: bgColor }}
                                    >
                                      {numScore.toFixed(3)}
                                    </span>
                                  )
                                })()}
                              </div>
                              <div className="text-xs text-gray-500 leading-relaxed space-y-1">
                                <div className="line-clamp-4">
                                  <span className="font-semibold text-blue-600">Prompt:</span>{' '}
                                  {(() => {
                                    const value = example[selectedTable.prompt_col]
                                    const text = isConversation(value) ? '[Conversation]' : String(value || '')
                                    return text.length > 400 ? text.substring(0, 400) + '...' : text
                                  })()}
                                </div>
                                <div className="line-clamp-4">
                                  <span className="font-semibold text-green-600">Answer:</span>{' '}
                                  {(() => {
                                    const value = example[selectedTable.answer_col]
                                    const text = isConversation(value) ? '[Conversation]' : String(value || '')
                                    return text.length > 400 ? text.substring(0, 400) + '...' : text
                                  })()}
                                </div>
                                <div className="line-clamp-4">
                                  <span className="font-semibold text-purple-600">Prediction:</span>{' '}
                                  {(() => {
                                    const value = example[selectedTable.pred_col]
                                    const text = isConversation(value) ? '[Conversation]' : String(value || '')
                                    return text.length > 400 ? text.substring(0, 400) + '...' : text
                                  })()}
                                </div>
                              </div>
                            </div>
                          ))}
                          </div>
                        ) : (
                          <div className="text-center text-gray-500 py-12 w-full">
                            <p>No examples match the active slices.</p>
                            <button
                              onClick={() => setActiveSlices(new Set())}
                              className="mt-2 px-3 py-1 text-sm text-blue-600 hover:text-blue-800 underline"
                            >
                              Clear slice filters
                            </button>
                          </div>
                        )}
                      </div>
                    ) : selectedTable ? (
                      <div className="text-center text-gray-500 py-12 w-full">
                        No table data available for this step.
                      </div>
                    ) : (
                      <div className="text-center text-gray-500 py-12 w-full">
                        Select a table step to view data.
                      </div>
                    )}
                  </div>
                ) : (
                  /* Table Focus Mode */
                  <div className="flex-1 flex flex-col w-full overflow-hidden">
                    <div className="flex items-center justify-between mb-4 pb-4 pt-6 border-b border-gray-300 px-6 w-full flex-shrink-0">
                      <button 
                        onClick={() => setSelectedTableExample(null)}
                        className="px-4 py-2 bg-gray-100 border border-gray-300 rounded cursor-pointer text-sm hover:bg-gray-200"
                      >← Back to Table</button>
                      <h2 className="text-xl font-semibold text-gray-800">
                        Table Example - Step {selectedTable.step}
                      </h2>
                      <div className="flex items-center gap-2">
                        <button 
                          onClick={() => navigateTableExample('prev')}
                          className="px-3 py-1 bg-gray-100 border border-gray-300 rounded text-sm hover:bg-gray-200"
                          title="Previous (←)"
                        >←</button>
                        <span className="text-sm text-gray-600">
                          {getFilteredTableData().findIndex(ex => JSON.stringify(ex) === JSON.stringify(selectedTableExample)) + 1} of {getFilteredTableData().length}
                        </span>
                        <button 
                          onClick={() => navigateTableExample('next')}
                          className="px-3 py-1 bg-gray-100 border border-gray-300 rounded text-sm hover:bg-gray-200"
                          title="Next (→)"
                        >→</button>
                      </div>
                    </div>
                    <div className="flex-1 overflow-y-auto px-6 w-full">
                      <div className="space-y-6">
                        {/* Prompt - can be either string or conversation */}
                        {(() => {
                          const promptValue = selectedTableExample[selectedTable.prompt_col]
                          if (isConversation(promptValue)) {
                            return renderConversation(promptValue, 'Prompt (Conversation)', 'bg-blue-50', 'border-blue-200', 'text-blue-800')
                          } else {
                            return (
                              <div className="p-4 rounded-lg bg-blue-50 border border-blue-200">
                                <div className="font-bold text-sm text-blue-800 mb-2">Prompt</div>
                                <div className="break-words leading-relaxed text-blue-900 prose prose-sm max-w-none">
                                  <ReactMarkdown>{String(promptValue || 'No prompt')}</ReactMarkdown>
                                </div>
                              </div>
                            )
                          }
                        })()}
                        {/* Expected Answer - can be either string or conversation */}
                        {(() => {
                          const answerValue = selectedTableExample[selectedTable.answer_col]
                          if (isConversation(answerValue)) {
                            return renderConversation(answerValue, 'Expected Answer (Conversation)', 'bg-green-50', 'border-green-200', 'text-green-800')
                          } else {
                            return (
                              <div className="p-4 rounded-lg bg-green-50 border border-green-200">
                                <div className="font-bold text-sm text-green-800 mb-2">Expected Answer</div>
                                <div className="break-words leading-relaxed text-green-900 prose prose-sm max-w-none">
                                  <ReactMarkdown>{String(answerValue || 'No answer')}</ReactMarkdown>
                                </div>
                              </div>
                            )
                          }
                        })()}
                        {/* Prediction - can be either string or conversation */}
                        {(() => {
                          const predValue = selectedTableExample[selectedTable.pred_col]
                          if (isConversation(predValue)) {
                            return renderConversation(predValue, 'Prediction (Conversation)', 'bg-purple-50', 'border-purple-200', 'text-purple-800')
                          } else {
                            return (
                              <div className="p-4 rounded-lg bg-purple-50 border border-purple-200">
                                <div className="font-bold text-sm text-purple-800 mb-2">Prediction</div>
                                <div className="break-words leading-relaxed text-purple-900 prose prose-sm max-w-none">
                                  <ReactMarkdown>{String(predValue || 'No prediction')}</ReactMarkdown>
                                </div>
                              </div>
                            )
                          }
                        })()}
                        <div className="p-4 rounded-lg bg-gray-50 border border-gray-200">
                          <div className="font-bold text-sm text-gray-800 mb-2">Score</div>
                          <div>
                            {(() => {
                              const score = selectedTableExample[selectedTable.score_col]
                              if (score === null || score === undefined || score === '') {
                                return (
                                  <span className="px-3 py-1 text-sm rounded-full bg-gray-200 text-gray-600">
                                    N/A
                                  </span>
                                )
                              }
                              
                              // Handle boolean values
                              if (typeof score === 'boolean' || score === 'true' || score === 'false') {
                                const isTrue = score === true || score === 'true'
                                return (
                                  <span className={`px-3 py-1 text-sm rounded-full font-medium text-white ${
                                    isTrue ? 'bg-green-500' : 'bg-red-500'
                                  }`}>
                                    {isTrue ? 'CORRECT' : 'INCORRECT'}
                                  </span>
                                )
                              }
                              
                              const numScore = parseFloat(score)
                              if (isNaN(numScore)) {
                                return (
                                  <span className="px-3 py-1 text-sm rounded-full bg-gray-200 text-gray-600">
                                    N/A
                                  </span>
                                )
                              }
                              
                              // Create gradient from red (0) to yellow (0.5) to green (1)
                              let bgColor, textColor
                              if (numScore < 0.5) {
                                // Red to yellow gradient
                                const intensity = numScore * 2 // 0 to 1
                                bgColor = `rgb(${255}, ${Math.round(165 + (255 - 165) * intensity)}, 0)`
                                textColor = 'text-white'
                              } else {
                                // Yellow to green gradient
                                const intensity = (numScore - 0.5) * 2 // 0 to 1
                                bgColor = `rgb(${Math.round(255 - 255 * intensity)}, 255, 0)`
                                textColor = intensity > 0.3 ? 'text-white' : 'text-black'
                              }
                              
                              return (
                                <span 
                                  className={`px-3 py-1 text-sm rounded-full font-medium ${textColor}`}
                                  style={{ backgroundColor: bgColor }}
                                >
                                  {numScore.toFixed(3)}
                                </span>
                              )
                            })()}
                          </div>
                        </div>
                        
                        {/* Full Row Data */}
                        <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
                          <div className="font-bold text-sm text-slate-800 mb-2">Full Row Data</div>
                          <div className="space-y-2">
                            {Object.entries(selectedTableExample).map(([key, value]) => (
                              <div key={key} className="border-b border-slate-200 pb-2 last:border-b-0 last:pb-0">
                                <div className="font-medium text-xs text-slate-700 mb-1">{key}</div>
                                <div className="text-xs text-slate-900">
                                  {isConversation(value) ? (
                                    <div className="space-y-2">
                                      {value.map((message, index) => {
                                        if (!message || typeof message !== 'object') {
                                          console.warn('Invalid message in Full Row Data conversation:', message)
                                          return null
                                        }
                                        return (
                                          <div key={index} className={`p-2 rounded border ${
                                            message.role === 'user' 
                                              ? 'bg-blue-50 border-blue-200' 
                                              : message.role === 'system'
                                                ? 'bg-yellow-50 border-yellow-200'
                                                : 'bg-green-50 border-green-200'
                                          }`}>
                                            <div className={`font-semibold text-xs ${
                                              message.role === 'user' ? 'text-blue-800' : message.role === 'system' ? 'text-yellow-800' : 'text-green-800'
                                            }`}>
                                              {String(message.role || 'unknown')}
                                            </div>
                                            <div className="mt-1 text-xs break-all">
                                              {String(message.content || '')}
                                            </div>
                                          </div>
                                        )
                                      }).filter(Boolean)}
                                    </div>
                                  ) : (
                                    <div className="break-all">
                                      {(() => {
                                        try {
                                          if (typeof value === 'object' && value !== null) {
                                            return JSON.stringify(value, null, 2)
                                          } else {
                                            return String(value || '')
                                          }
                                        } catch (error) {
                                          console.warn('Error rendering value:', key, value, error)
                                          return `[Error rendering ${typeof value}]`
                                        }
                                      })()}
                                    </div>
                                  )}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Loading State */}
            {loadingDashboard && (
              <div className="flex flex-col items-center justify-center py-12 text-center text-gray-600">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-4"></div>
                <p>Loading run from W&B...</p>
              </div>
            )}
          </div>
        ) : (
          loadingDashboard ? (
            <div className="flex flex-col items-center justify-center text-center text-gray-600 flex-1 p-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mb-4"></div>
              <h2 className="text-xl font-semibold mb-2 text-gray-800">Loading run from W&B...</h2>
              <p>Loading dashboard data and metrics.</p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center text-center text-gray-600 flex-1 p-4">
              <h2 className="text-xl font-semibold mb-2 text-gray-800">Select a run to analyze</h2>
              <p>Choose a dashboard and click on a W&B run to begin analysis.</p>
            </div>
          )
        )}
      </div>
    </div>
  )
}

export default TrainingPage