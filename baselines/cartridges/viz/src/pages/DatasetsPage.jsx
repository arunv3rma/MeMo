import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'

function DatasetsPage() {
  const [datasets, setDatasets] = useState([])
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [examples, setExamples] = useState([])
  const [totalExamples, setTotalExamples] = useState(0)
  const [selectedExample, setSelectedExample] = useState(null)
  const [selectedExampleWithLogprobs, setSelectedExampleWithLogprobs] = useState(null)
  const [loadingLogprobs, setLoadingLogprobs] = useState(false)
  const [showTokenPanel, setShowTokenPanel] = useState(false)
  const [selectedTokens, setSelectedTokens] = useState(null)
  const [hoveredTokenIndex, setHoveredTokenIndex] = useState(null)
  const [panelPosition, setPanelPosition] = useState({ x: 0, y: 0 })
  const [outputDir, setOutputDir] = useState('')
  const [tokenizerName, setTokenizerName] = useState('meta-llama/Llama-3.2-3B-Instruct')
  const [systemPromptExpanded, setSystemPromptExpanded] = useState(false)
  const [currentPage, setCurrentPage] = useState(0)
  const [examplesPerPage] = useState(128)
  const [loadingDatasetPath, setLoadingDatasetPath] = useState(null)
  const [datasetError, setDatasetError] = useState(null)
  const [configData, setConfigData] = useState(null)
  const [loadingDatasets, setLoadingDatasets] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchFields, setSearchFields] = useState({
    messages: true,
    system_prompt: false,
    metadata: false
  })
  const [isScrolled, setIsScrolled] = useState(false)
  const [copySuccess, setCopySuccess] = useState({})
  const [collapsedMessages, setCollapsedMessages] = useState({})
  const [currentAbortController, setCurrentAbortController] = useState(null)

  // Navigate examples (similar to table navigation)
  const navigateExample = async (direction) => {
    if (!selectedExample || examples.length === 0) return
    
    const currentIndex = examples.findIndex(ex => ex === selectedExample)
    let newIndex
    
    if (direction === 'next') {
      newIndex = (currentIndex + 1) % examples.length
    } else {
      newIndex = (currentIndex - 1 + examples.length) % examples.length
    }
    
    // Immediately show the new example
    setSelectedExample(examples[newIndex])
    setSelectedExampleWithLogprobs(null) // Clear previous detailed data
    setCollapsedMessages({}) // Reset collapsed state
    setCopySuccess({}) // Reset copy success state
    
    // Then fetch detailed example with logprobs in background
    const globalIndex = (currentPage * examplesPerPage) + newIndex
    console.log('Navigating to example at global index:', globalIndex)
    const detailedExample = await fetchExampleWithLogprobs(globalIndex)
    if (detailedExample) {
      setSelectedExampleWithLogprobs(detailedExample)
    }
  }

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (!selectedExample) return
      
      if (event.key === 'ArrowRight') {
        event.preventDefault()
        navigateExample('next')
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault()
        navigateExample('prev')
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedExample, examples])

  // Dataset discovery function
  const discoverDatasets = async () => {
    try {
      setLoadingDatasets(true)
      const response = await fetch('/api/datasets')
      const data = await response.json()
      // Ensure datasets are sorted by relative path (reverse alphabetical)
      const sortedData = data.sort((a, b) => b.relative_path.localeCompare(a.relative_path))
      setDatasets(sortedData)
      
      // Auto-select the most recent dataset (first in sorted list) if no dataset is currently selected
      if (sortedData.length > 0 && !selectedDataset) {
        console.log('Auto-selecting most recent dataset:', sortedData[0].path)
        selectDataset(sortedData[0].path)
      }
    } catch (error) {
      console.error('Failed to discover datasets:', error)
    } finally {
      setLoadingDatasets(false)
    }
  }

  // Dataset discovery
  useEffect(() => {
    discoverDatasets()
  }, [outputDir])

  // Cleanup abort controller on unmount
  useEffect(() => {
    return () => {
      if (currentAbortController) {
        currentAbortController.abort()
      }
    }
  }, [currentAbortController])

  const selectDataset = async (datasetPath) => {
    // Cancel any pending logprobs request when switching datasets
    if (currentAbortController) {
      currentAbortController.abort()
      setCurrentAbortController(null)
    }
    
    // Immediately set the selected dataset and show the path
    setSelectedDataset(datasetPath)
    setExamples([])
    setSelectedExample(null)
    setCurrentPage(0)
    setLoadingDatasetPath(datasetPath)
    setDatasetError(null)
    setConfigData(null)
    
    try {
      // First, load dataset metadata quickly
      const infoResponse = await fetch(`/api/dataset/${encodeURIComponent(datasetPath)}/info`)
      const info = await infoResponse.json()
      setTotalExamples(info.total_count)
      
      // Reset search and load the first page of examples
      setSearchQuery('')
      await loadDatasetWithSearch(0, datasetPath)
      
      // Also automatically load the config
      try {
        const configResponse = await fetch('/api/dataset/config', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            dataset_path: datasetPath
          })
        })
        const configData = await configResponse.json()
        if (configData.config) {
          setConfigData(configData.config)
        }
      } catch (configError) {
        console.log('No config found for dataset:', configError)
      }
    } catch (error) {
      console.error('Failed to load dataset:', error)
      setDatasetError(`Failed to load dataset: ${error.message}`)
      setExamples([])
      setTotalExamples(0)
    } finally {
      setLoadingDatasetPath(null)
    }
  }

  const loadDatasetWithSearch = async (page = currentPage, datasetPath = null) => {
    const dataset = datasetPath || selectedDataset
    if (!dataset) return

    setLoadingDatasetPath(dataset)
    setDatasetError(null)
    
    try {
      const searchParams = new URLSearchParams({
        page: page.toString(),
        page_size: examplesPerPage.toString(),
        search: searchQuery || '',
        search_messages: searchFields.messages.toString(),
        search_system_prompt: searchFields.system_prompt.toString(),
        search_metadata: searchFields.metadata.toString()
      })

      const response = await fetch(`/api/dataset/${encodeURIComponent(dataset)}?${searchParams}`)
      const data = await response.json()
      
      setExamples(data.examples)
      setTotalExamples(data.total_count)
      setCurrentPage(page)
    } catch (error) {
      console.error('Failed to load dataset with search:', error)
      setDatasetError(`Failed to load dataset: ${error.message}`)
    } finally {
      setLoadingDatasetPath(null)
    }
  }

  const handleSearch = () => {
    setCurrentPage(0)
    loadDatasetWithSearch(0)
  }

  const handleCopyContent = (content, messageIndex) => {
    navigator.clipboard.writeText(content).then(() => {
      setCopySuccess(prev => ({...prev, [messageIndex]: true}))
      setTimeout(() => {
        setCopySuccess(prev => ({...prev, [messageIndex]: false}))
      }, 2000)
    })
  }

  const toggleMessageCollapse = (messageIndex) => {
    setCollapsedMessages(prev => ({
      ...prev,
      [messageIndex]: !prev[messageIndex]
    }))
  }

  // Fetch detailed example with logprobs
  const fetchExampleWithLogprobs = async (exampleIndex) => {
    if (!selectedDataset) return null

    // Cancel any existing request
    if (currentAbortController) {
      currentAbortController.abort()
    }

    // Create new abort controller for this request
    const abortController = new AbortController()
    setCurrentAbortController(abortController)

    setLoadingLogprobs(true)
    try {
      console.log('Fetching example with logprobs, index:', exampleIndex, 'dataset:', selectedDataset)
      const response = await fetch('/api/dataset/example', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          dataset_path: selectedDataset,
          example_index: exampleIndex
        }),
        signal: abortController.signal
      })
      const data = await response.json()
      
      console.log('Received detailed example data:', data)
      
      if (data.error) {
        console.error('Error fetching example with logprobs:', data.error)
        return null
      }
      
      if (data.example && data.example.messages) {
        console.log('Example messages with logprobs:', data.example.messages.map(m => ({
          role: m.role,
          has_token_ids: !!m.token_ids,
          token_ids_length: m.token_ids?.length,
          has_top_logprobs: !!m.top_logprobs,
          top_logprobs_structure: m.top_logprobs
        })))
      }
      
      // Clear the abort controller since request completed successfully
      if (currentAbortController === abortController) {
        setCurrentAbortController(null)
      }
      
      return data.example
    } catch (error) {
      // Only log error if it wasn't an abort
      if (!abortController.signal.aborted) {
        console.error('Failed to fetch example with logprobs:', error)
      }
      return null
    } finally {
      // Only clear loading state if this request wasn't aborted
      if (!abortController.signal.aborted) {
        setLoadingLogprobs(false)
      }
    }
  }

  // Token decoding function
  const decodeTokens = async (tokenIds, topLogprobs = null) => {
    try {
      
      if (!selectedDataset) {
        console.error('No selectedDataset available for token decoding')
        return tokenIds.map((id, idx) => ({ id, token: `[${id}]`, prob: 0 }))
      }
      
      const response = await fetch('/api/decode-tokens', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          token_ids: tokenIds,
          dataset_path: selectedDataset
        })
      })
      
      
      const data = await response.json()
      
      if (data.error) {
        return tokenIds.map((id, idx) => ({ id, token: `[${id}]`, prob: 0 }))
      }
      
      // Combine token IDs with decoded tokens and probabilities
      return tokenIds.map((id, idx) => {
        let token = data.decoded_tokens[idx] || `[${id}]`
        // Handle empty or problematic tokens
        if (!token || token.trim() === '' || token === '�') {
          token = `[${id}]`
        }
        
        // Try to get probability from top_logprobs if available
        let prob = null // No default probability
        console.log('topLogprobs for token', idx, ':', topLogprobs)
        
        if (topLogprobs && Array.isArray(topLogprobs) && topLogprobs[idx]) {
          // topLogprobs is now a list of lists where each index corresponds to token position
          const tokenLogprobEntries = topLogprobs[idx]
          
          if (Array.isArray(tokenLogprobEntries) && tokenLogprobEntries.length > 0) {
            // Find the entry that matches this token_id, or use the first one
            const matchingEntry = tokenLogprobEntries.find(entry => entry.token_id === id) || tokenLogprobEntries[0]
            
            if (matchingEntry && typeof matchingEntry.logprob === 'number') {
              prob = Math.exp(matchingEntry.logprob) // Convert log prob to probability
              console.log(`Token ${id} at pos ${idx} logprob: ${matchingEntry.logprob} -> prob: ${prob}`)
            }
          }
        }
        
        return {
          id,
          token,
          prob: prob ? Math.min(Math.max(prob, 0.1), 1.0) : null // Clamp between 0.1 and 1.0, or null
        }
      })
    } catch (error) {
      console.error('Failed to decode tokens:', error)
      return tokenIds.map((id, idx) => ({ id, token: `[${id}]`, prob: 0 }))
    }
  }

  const getColorForProbability = (prob) => {
    // Convert probability to hue (green = high prob, red = low prob)
    const hue = prob * 120 // 0 to 120 degrees
    const saturation = 70
    const lightness = 85
    return {
      backgroundColor: `hsl(${hue}, ${saturation}%, ${lightness}%)`,
      borderColor: `hsl(${hue}, ${saturation}%, ${lightness - 20}%)`
    }
  }

  // Token Hover Panel Component - shows top-k alternatives for a single token
  const TokenHoverPanel = ({ tokenAlternatives, position, show, tokenIndex }) => {
    if (!tokenAlternatives || !show) return null

    return (
      <div 
        className="fixed bg-white border border-gray-200 rounded-lg shadow-xl p-4 z-50 max-w-sm"
        style={{
          left: `${position.x}px`,
          top: `${position.y}px`,
          transform: 'translate(-50%, -100%)',
          marginTop: '-8px'
        }}
      >
        <div className="text-sm font-medium text-gray-800 mb-3">
          Token #{tokenIndex} Alternatives
        </div>
        <div className="space-y-3 max-h-64 overflow-y-auto">
          {tokenAlternatives.map((alternative, idx) => {
            const prob = Math.exp(alternative.logprob)
            return (
              <div key={idx} className="border-b border-gray-100 last:border-b-0 pb-2 last:pb-0">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-mono text-sm bg-gray-100 px-1 rounded">
                    {(() => {
                      // Use the token_str from the backend if available
                      let displayToken = alternative.token_str || `[${alternative.token_id}]`
                      if (displayToken === ' ') return '␣'
                      if (displayToken === '\n') return '⏎'
                      if (displayToken === '\t') return '⭾'
                      return displayToken
                    })()}
                  </span>
                  <span className="text-xs font-medium text-gray-600">
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
                
                {/* Mini probability bar */}
                <div className="w-full bg-gray-200 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="h-full transition-all duration-300 ease-out rounded-full"
                    style={{
                      width: `${prob * 100}%`,
                      backgroundColor: `hsl(${prob * 120}, 70%, 55%)`
                    }}
                  />
                </div>
                
                <div className="text-xs text-gray-400 mt-1">
                  logprob: {alternative.logprob.toFixed(3)}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    )
  }

  const TokenVisualization = ({ tokenIds, message }) => {
    // Don't do any async loading - just use the token data directly
    if (!tokenIds || tokenIds.length === 0) {
      return <div className="text-sm text-gray-500">No tokens available</div>
    }

    // Create token display data directly from the message data
    const tokens = tokenIds.map((id, idx) => {
      // Try to get probability from top_logprobs if available
      let prob = null
      
      if (message.top_logprobs && Array.isArray(message.top_logprobs) && message.top_logprobs[idx]) {
        const tokenLogprobEntries = message.top_logprobs[idx]
        
        if (Array.isArray(tokenLogprobEntries) && tokenLogprobEntries.length > 0) {
          // Find the entry that matches this token_id, or use the first one
          const matchingEntry = tokenLogprobEntries.find(entry => entry.token_id === id) || tokenLogprobEntries[0]
          
          if (matchingEntry && typeof matchingEntry.logprob === 'number') {
            prob = Math.exp(matchingEntry.logprob) // Convert log prob to probability
          }
        }
      }
      
      // Get the actual token string from the backend data
      const tokenStr = message.token_strs && message.token_strs[idx] ? message.token_strs[idx] : `[${id}]`
      
      return {
        id,
        token: tokenStr,
        prob: prob ? Math.min(Math.max(prob, 0.1), 1.0) : null
      }
    })

    return (
      <div className="mt-4 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-semibold mb-2 text-gray-800">Tokens</h4>
        <div className="relative token-container">
          <div className="font-mono text-sm leading-relaxed">
            {tokens.map((token, idx) => {
              const colors = token.prob !== null 
                ? getColorForProbability(token.prob)
                : { backgroundColor: '#e5e7eb', borderColor: '#9ca3af' } // Grey for no probability
              
              // Get alternatives for this token position from top_logprobs
              const tokenAlternatives = message.top_logprobs && message.top_logprobs[idx] ? message.top_logprobs[idx] : []
              
              return (
                <span
                  key={idx}
                  className="inline-block mr-1 mb-1 px-1 py-0.5 rounded border transition-all hover:scale-105 relative cursor-pointer"
                  style={colors}
                  onMouseEnter={(e) => {
                    if (tokenAlternatives.length > 0) {
                      setSelectedTokens(tokenAlternatives)
                      setHoveredTokenIndex(idx)
                      setShowTokenPanel(true)
                      const rect = e.currentTarget.getBoundingClientRect()
                      setPanelPosition({
                        x: rect.left + rect.width / 2,
                        y: rect.top
                      })
                    }
                  }}
                  onMouseLeave={() => {
                    setShowTokenPanel(false)
                    setSelectedTokens(null)
                    setHoveredTokenIndex(null)
                  }}
                >
                  {(() => {
                    let displayToken = token.token
                    // Handle special characters and whitespace
                    if (displayToken === ' ') {
                      return '␣' // Visible space character
                    } else if (displayToken === '\n') {
                      return '⏎' // Visible newline
                    } else if (displayToken === '\t') {
                      return '⭾' // Visible tab
                    } else if (displayToken.trim() === '' && displayToken.length > 0) {
                      return '·'.repeat(displayToken.length) // Visible whitespace
                    } else {
                      return displayToken.replace(/\n/g, '\\n').replace(/\t/g, '\\t')
                    }
                  })()}
                </span>
              )
            })}
          </div>
        </div>
      </div>
    )
  }

  // Filter examples based on search
  const filteredExamples = examples.filter(example => {
    if (!searchQuery.trim()) return true
    
    const query = searchQuery.toLowerCase()
    return example.messages.some(msg => 
      msg.content.toLowerCase().includes(query)
    )
  })


  return (
    <>
      {/* Token Hover Panel */}
      <TokenHoverPanel 
        tokenAlternatives={selectedTokens} 
        position={panelPosition} 
        show={showTokenPanel}
        tokenIndex={hoveredTokenIndex}
      />
      
      <div className="flex h-full w-full font-sans overflow-hidden">
      <div className="w-96 bg-gray-100 border-r border-gray-300 flex flex-col flex-shrink-0 overflow-hidden">
        {/* Navigation Tabs */}
        <div className="p-4 border-b border-gray-300">
          <div className="flex space-x-1">
            <div className="px-3 py-2 text-sm font-medium text-blue-600 bg-blue-100 rounded">
              Datasets
            </div>
            <a
              href="/training"
              className="px-3 py-2 text-sm font-medium text-gray-600 hover:text-gray-800 hover:bg-gray-200 rounded transition-colors"
            >
              Training
            </a>
          </div>
        </div>

        {/* Refresh Button */}
        <div className="p-4 border-b border-gray-300">
          <button
            onClick={discoverDatasets}
            disabled={loadingDatasets}
            className={`w-full px-4 py-2 rounded text-sm font-medium transition-all duration-200 ${
              loadingDatasets
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            {loadingDatasets ? 'Refreshing...' : 'Refresh Datasets'}
          </button>
        </div>

        {/* Dataset List */}
        <div className="flex-1 overflow-y-auto p-4">
          <div className="flex flex-col gap-2">
            {loadingDatasets ? (
              <div className="flex flex-col items-center justify-center py-8 text-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600 mb-3"></div>
                <div className="text-sm text-gray-600">Loading datasets...</div>
              </div>
            ) : (
              <>
                <h2 className="text-sm font-medium text-gray-700 mb-2">
                  Available Datasets ({datasets.length})
                </h2>
                {datasets.map((dataset, index) => (
                  <div
                    key={index}
                    className={`p-3 rounded-lg cursor-pointer transition-all duration-200 border ${
                      selectedDataset === dataset.path
                        ? 'bg-purple-100 border-purple-300 shadow-sm'
                        : 'bg-white border-gray-200 hover:bg-gray-50 hover:border-gray-300'
                    }`}
                    onClick={() => selectDataset(dataset.path)}
                  >
                    <div className="font-medium text-sm text-gray-800 mb-1 break-words">
                      {dataset.name}
                    </div>
                    <div className="text-xs text-gray-500 break-words mb-1">
                      {dataset.relative_path}
                    </div>
                    <div className="text-xs text-gray-400">
                      {dataset.size.toFixed(2)} GB
                    </div>
                  </div>
                ))}
              </>
            )}
          </div>
        </div>
      </div>
      
      <div className="flex-1 flex flex-col min-w-0 w-full overflow-hidden">
        {!selectedDataset ? (
          <div className="flex flex-col items-center justify-center text-center text-gray-600 flex-1 p-4">
            <h2 className="text-xl font-semibold mb-2 text-gray-800">Select a dataset to explore</h2>
            <p>Choose a dataset from the sidebar to begin exploring training examples.</p>
          </div>
        ) : !selectedExample ? (
          <>
            {/* Search and Filter Controls */}
            <div className="bg-white p-4 border-b border-gray-200">
              <div className="flex flex-col gap-4">
                {/* Search Input */}
                <div className="flex gap-2">
                  <input
                    type="text"
                    placeholder="Search examples..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="flex-1 p-2 border border-gray-300 rounded text-sm"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        handleSearch()
                      }
                    }}
                  />
                  <button
                    onClick={handleSearch}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded transition-colors"
                  >
                    Search
                  </button>
                </div>

                {/* Search Field Checkboxes */}
                <div className="flex gap-4 text-sm">
                  <label className="flex items-center gap-1">
                    <input
                      type="checkbox"
                      checked={searchFields.messages}
                      onChange={(e) => setSearchFields({...searchFields, messages: e.target.checked})}
                    />
                    Messages
                  </label>
                  <label className="flex items-center gap-1">
                    <input
                      type="checkbox"
                      checked={searchFields.system_prompt}
                      onChange={(e) => setSearchFields({...searchFields, system_prompt: e.target.checked})}
                    />
                    System Prompt
                  </label>
                  <label className="flex items-center gap-1">
                    <input
                      type="checkbox"
                      checked={searchFields.metadata}
                      onChange={(e) => setSearchFields({...searchFields, metadata: e.target.checked})}
                    />
                    Metadata
                  </label>
                </div>

                {/* Dataset Info */}
                <div className="text-sm text-gray-600">
                  <strong>{selectedDataset.split('/').pop()}</strong>
                  {searchQuery && (
                    <span> - Showing {totalExamples} results for "{searchQuery}"</span>
                  )}
                  {!searchQuery && (
                    <span> - {totalExamples} total examples</span>
                  )}
                </div>
              </div>
            </div>

            {/* Examples Grid */}
            <div 
              className="flex-1 overflow-y-auto p-4"
              onScroll={(e) => {
                const scrolled = e.target.scrollTop > 50
                if (scrolled !== isScrolled) {
                  setIsScrolled(scrolled)
                }
              }}
            >
              {datasetError ? (
                <div className="flex flex-col items-center justify-center py-12 text-center text-red-600">
                  <div className="text-red-500 mb-4">⚠️</div>
                  <h3 className="font-semibold mb-2">Error loading dataset</h3>
                  <p className="text-sm text-gray-600">{datasetError}</p>
                  <button 
                    onClick={() => selectDataset(selectedDataset)}
                    className="mt-4 px-4 py-2 bg-red-100 hover:bg-red-200 text-red-800 rounded text-sm"
                  >
                    Retry
                  </button>
                </div>
              ) : loadingDatasetPath && examples.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-center text-gray-600">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-4"></div>
                  <p>Loading examples...</p>
                </div>
              ) : filteredExamples.length > 0 ? (
                <>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 w-full">
                    {filteredExamples.map((example, index) => (
                      <div
                        key={index}
                        className="border border-gray-300 rounded-lg p-4 cursor-pointer transition-all hover:-translate-y-1 hover:shadow-lg"
                        onClick={async () => {
                          // Immediately show the basic example
                          setSelectedExample(example)
                          setSelectedExampleWithLogprobs(null) // Clear previous detailed data
                          setCollapsedMessages({}) // Reset collapsed state
                          setCopySuccess({}) // Reset copy success state
                          
                          // Then fetch detailed data with logprobs in the background
                          const globalIndex = (currentPage * examplesPerPage) + index
                          console.log('Fetching detailed example at global index:', globalIndex)
                          const detailedExample = await fetchExampleWithLogprobs(globalIndex)
                          if (detailedExample) {
                            setSelectedExampleWithLogprobs(detailedExample)
                          }
                        }}
                      >
                        <div className="text-sm text-gray-600 mb-2">
                          Example {(currentPage * examplesPerPage) + index + 1}
                        </div>
                        <div className="text-xs text-gray-500 leading-relaxed space-y-1">
                          {example.system_prompt && (
                            <div>
                              <span className="font-semibold text-yellow-600">System:</span>{' '}
                              {example.system_prompt.substring(0, 60)}...
                            </div>
                          )}
                          {example.messages.slice(0, 2).map((message, msgIndex) => (
                            <div key={msgIndex}>
                              <span className={`font-semibold ${
                                message.role === 'user' ? 'text-blue-600' : message.role === 'system' ? 'text-yellow-600' : 'text-green-600'
                              }`}>
                                {message.role === 'user' ? 'User:' : message.role === 'system' ? 'System:' : 'Assistant:'}
                              </span>{' '}
                              {message.content.substring(0, 80)}...
                            </div>
                          ))}
                          {example.messages.length > 2 && (
                            <div className="text-gray-400 text-xs">
                              +{example.messages.length - 2} more messages
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  {/* Pagination Controls */}
                  {totalExamples > examplesPerPage && (
                    <div className="flex justify-center items-center mt-6 gap-2">
                      <button 
                        onClick={() => loadDatasetWithSearch(Math.max(0, currentPage - 1))}
                        disabled={currentPage === 0}
                        className="px-3 py-1 bg-gray-100 border border-gray-300 rounded text-sm hover:bg-gray-200 disabled:opacity-50"
                      >
                        Previous
                      </button>
                      <span className="px-3 py-1 text-sm text-gray-600">
                        Page {currentPage + 1} of {Math.ceil(totalExamples / examplesPerPage)}
                      </span>
                      <button 
                        onClick={() => loadDatasetWithSearch(currentPage + 1)}
                        disabled={(currentPage + 1) * examplesPerPage >= totalExamples}
                        className="px-3 py-1 bg-gray-100 border border-gray-300 rounded text-sm hover:bg-gray-200 disabled:opacity-50"
                      >
                        Next
                      </button>
                    </div>
                  )}
                </>
              ) : (
                <div className="text-center text-gray-500 py-12">
                  {searchQuery ? `No examples found for "${searchQuery}"` : 'No examples found in this dataset.'}
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="h-full flex flex-col">
            <div className="flex items-center justify-between mb-4 pb-4 pt-6 border-b border-gray-300 px-6">
              <div className="flex items-center gap-4">
                <button 
                  onClick={() => {
                    // Cancel any pending logprobs request when going back to examples list
                    if (currentAbortController) {
                      currentAbortController.abort()
                      setCurrentAbortController(null)
                                    }
                    setSelectedExample(null)
                    setSelectedExampleWithLogprobs(null)
                    setCollapsedMessages({}) // Reset collapsed state
                    setCopySuccess({}) // Reset copy success state
                  }}
                  className="px-4 py-2 bg-gray-100 border border-gray-300 rounded cursor-pointer text-sm hover:bg-gray-200"
                >
                  ← Back to Examples
                </button>
                <div className="text-sm text-gray-600">
                  <strong>{selectedDataset.split('/').pop()}</strong> - Example {examples.findIndex(ex => ex === selectedExample) + 1 + (currentPage * examplesPerPage)} of {totalExamples}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button 
                  onClick={() => navigateExample('prev')}
                  className="px-3 py-1 bg-gray-100 border border-gray-300 rounded text-sm hover:bg-gray-200"
                  title="Previous (←)"
                >←</button>
                <span className="text-sm text-gray-600">
                  {examples.findIndex(ex => ex === selectedExample) + 1} of {examples.length}
                </span>
                <button 
                  onClick={() => navigateExample('next')}
                  className="px-3 py-1 bg-gray-100 border border-gray-300 rounded text-sm hover:bg-gray-200"
                  title="Next (→)"
                >→</button>
              </div>
            </div>
            <div className="flex-1 overflow-y-auto px-6">
              <div className="flex flex-col gap-4">
              {selectedExample.system_prompt && (
                <div className="p-4 rounded-lg bg-yellow-50 border border-yellow-200">
                  <div className="flex items-center justify-between mb-2">
                    <div className="font-bold text-sm text-yellow-800">System Prompt</div>
                    {selectedExample.system_prompt.length > 300 && (
                      <button
                        onClick={() => setSystemPromptExpanded(!systemPromptExpanded)}
                        className="text-xs px-2 py-1 bg-yellow-200 hover:bg-yellow-300 text-yellow-800 rounded"
                      >
                        {systemPromptExpanded ? 'Collapse' : 'Expand'}
                      </button>
                    )}
                  </div>
                  <div className="break-words leading-relaxed text-yellow-900 prose prose-sm max-w-none">
                    <ReactMarkdown>
                      {systemPromptExpanded 
                        ? selectedExample.system_prompt 
                        : selectedExample.system_prompt.length > 300 
                          ? selectedExample.system_prompt.substring(0, 300) + '...'
                          : selectedExample.system_prompt
                      }
                    </ReactMarkdown>
                  </div>
                </div>
              )}

              {(selectedExampleWithLogprobs || selectedExample).messages.map((message, messageIndex) => (
                <div key={messageIndex} className={`p-4 rounded-lg border ${
                  message.role === 'user' 
                    ? 'bg-blue-50 border-blue-200' 
                    : message.role === 'system'
                      ? 'bg-yellow-50 border-yellow-200'
                      : 'bg-green-50 border-green-200'
                }`}>
                  <div className="flex items-center justify-between mb-2">
                    <div className={`font-bold text-sm ${
                      message.role === 'user' ? 'text-blue-800' : message.role === 'system' ? 'text-yellow-800' : 'text-green-800'
                    }`}>
                      {message.role === 'user' ? 'User' : message.role === 'system' ? 'System' : 'Assistant'}
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => toggleMessageCollapse(messageIndex)}
                        className={`text-xs px-2 py-1 rounded transition-all ${
                          message.role === 'user' 
                            ? 'bg-blue-200 hover:bg-blue-300 text-blue-800' 
                            : message.role === 'system'
                              ? 'bg-yellow-200 hover:bg-yellow-300 text-yellow-800'
                              : 'bg-green-200 hover:bg-green-300 text-green-800'
                        }`}
                      >
                        {collapsedMessages[messageIndex] ? 'Expand' : 'Collapse'}
                      </button>
                      <button
                        onClick={() => handleCopyContent(message.content, messageIndex)}
                        className={`text-xs px-2 py-1 rounded transition-all ${
                          copySuccess[messageIndex] 
                            ? 'bg-green-200 text-green-800' 
                            : message.role === 'user' 
                              ? 'bg-blue-200 hover:bg-blue-300 text-blue-800' 
                              : message.role === 'system'
                                ? 'bg-yellow-200 hover:bg-yellow-300 text-yellow-800'
                                : 'bg-green-200 hover:bg-green-300 text-green-800'
                        }`}
                      >
                        {copySuccess[messageIndex] ? '✓ Copied' : 'Copy'}
                      </button>
                    </div>
                  </div>
                  {!collapsedMessages[messageIndex] && (
                    <div className={`break-words leading-relaxed prose prose-sm max-w-none ${
                      message.role === 'user' ? 'text-blue-900' : message.role === 'system' ? 'text-yellow-900' : 'text-green-900'
                    }`}>
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    </div>
                  )}
                  
                  {/* Token visualization */}
                  {!collapsedMessages[messageIndex] && message.token_ids && message.token_ids.length > 0 && (
                    <div>
                      {/* Show loading state when we have basic message but waiting for logprobs */}
                      {/* Always show tokens, with loading state if waiting for probabilities */}
                      {!message.top_logprobs && loadingLogprobs ? (
                        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                          <h4 className="font-semibold mb-2 text-gray-800">Tokens</h4>
                          <div className="text-sm text-gray-500 flex items-center">
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                            Loading probabilities...
                          </div>
                        </div>
                      ) : (
                        <TokenVisualization tokenIds={message.token_ids} message={message} />
                      )}
                    </div>
                  )}
                </div>
              ))}

              {/* Metadata */}
              {selectedExample.metadata && Object.keys(selectedExample.metadata).length > 0 && (
                <div className="p-4 rounded-lg bg-gray-50 border border-gray-200">
                  <div className="font-bold text-sm text-gray-800 mb-2">Metadata</div>
                  <div className="text-sm text-gray-700">
                    <pre className="whitespace-pre-wrap font-mono text-xs bg-gray-100 p-2 rounded">
                      {JSON.stringify(selectedExample.metadata, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
              
              {/* Config Display */}
              {configData && (
                <div className="p-4 rounded-lg bg-purple-50 border border-purple-200">
                  <div className="font-bold text-sm text-purple-800 mb-2">Dataset Configuration</div>
                  <div className="text-sm text-purple-700">
                    <pre className="whitespace-pre-wrap font-mono text-xs bg-purple-100 p-2 rounded">
                      {JSON.stringify(configData, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
    </>
  )
}

export default DatasetsPage