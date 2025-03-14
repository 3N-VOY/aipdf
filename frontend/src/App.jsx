import React, { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import {
  FileText,
  Send,
  Upload,
  Bot,
  Loader,
  Trash2,
  Files,
} from "lucide-react";

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const BACKEND_URL = "http://localhost:8000"; // FastAPI backend URL

  const onDrop = useCallback(async (acceptedFiles) => {
    const pdfFile = acceptedFiles[0];
    if (pdfFile?.type === "application/pdf") {
      setFile(pdfFile);
      setLoading(true);

      try {
        const formData = new FormData();
        formData.append("file", pdfFile);

        const response = await fetch(`${BACKEND_URL}/upload`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) throw new Error("Failed to process PDF");

        setMessages((prev) => [
          ...prev,
          {
            type: "system",
            content: `Uploaded and processed: ${pdfFile.name}`,
          },
        ]);
      } catch (error) {
        setMessages((prev) => [
          ...prev,
          {
            type: "system",
            content: `Error processing PDF: ${error.message}`,
          },
        ]);
        setFile(null);
      } finally {
        setLoading(false);
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    multiple: false,
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim() || !file) return;

    setMessages((prev) => [
      ...prev,
      {
        type: "user",
        content: question,
      },
    ]);

    setLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) throw new Error("Failed to get answer");

      const { answer, context } = await response.json();
      setMessages((prev) => [
        ...prev,
        {
          type: "assistant",
          content: answer,
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          type: "system",
          content: `Error: ${error.message}`,
        },
      ]);
    } finally {
      setLoading(false);
      setQuestion("");
    }
  };

  const removeFile = () => {
    setFile(null);
    setMessages([]);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-gray-100 flex">
      {/* Sidebar */}
      <div
        className={`w-80 bg-gray-800 border-r border-gray-700 transition-all duration-300 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="p-4">
          <div className="flex items-center gap-2 mb-8">
            <Files className="h-6 w-6 text-blue-400" />
            <h2 className="text-xl font-semibold">Documents</h2>
          </div>

          {file ? (
            <div className="bg-gray-700 rounded-lg p-4 mb-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3 max-w-[85%]">
                  <FileText className="text-blue-400 flex-shrink-0" />
                  <span
                    className="font-medium truncate w-full"
                    title={file.name} // This adds a tooltip on hover
                  >
                    {file.name}
                  </span>
                </div>
                <button
                  onClick={removeFile}
                  className="text-gray-400 hover:text-red-400 transition-colors flex-shrink-0"
                >
                  <Trash2 className="h-5 w-5" />
                </button>
              </div>
              <div className="mt-2 text-sm text-gray-400">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </div>
            </div>
          ) : (
            <div className="text-gray-400 text-sm">
              No documents uploaded yet
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        <header className="text-center py-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent mb-2">
            PDF Q&A Assistant
          </h1>
          <p className="text-gray-400">
            Upload a PDF and ask questions about its content
          </p>
        </header>

        <div className="flex-1 max-w-4xl mx-auto w-full px-4 pb-4">
          {!file && (
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all
                ${
                  isDragActive
                    ? "border-blue-400 bg-gray-800/50"
                    : "border-gray-600 hover:border-gray-500 hover:bg-gray-800/30"
                }`}
            >
              <input {...getInputProps()} />
              <Upload className="mx-auto h-12 w-12 text-blue-400 mb-4" />
              <p className="text-gray-400">
                Drag & drop a PDF file here, or click to select one
              </p>
            </div>
          )}

          <div className="mt-6 bg-gray-800 rounded-lg shadow-xl border border-gray-700">
            <div className="h-[400px] overflow-y-auto p-4 border-b border-gray-700">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`mb-4 flex items-start gap-3 ${
                    message.type === "user" ? "justify-end" : ""
                  }`}
                >
                  {message.type === "assistant" && (
                    <Bot className="h-6 w-6 text-blue-400" />
                  )}
                  <div
                    className={`rounded-lg p-3 max-w-[80%] ${
                      message.type === "user"
                        ? "bg-blue-500 text-white"
                        : message.type === "system"
                        ? "bg-gray-700 text-gray-300"
                        : "bg-gray-700 text-gray-200"
                    }`}
                  >
                    {message.content}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="flex items-center gap-2 text-gray-400">
                  <Loader className="h-5 w-5 animate-spin" />
                  <span>Processing...</span>
                </div>
              )}
            </div>

            <form onSubmit={handleSubmit} className="p-4 flex gap-2">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask a question about the PDF..."
                className="flex-1 rounded-lg bg-gray-700 border border-gray-600 px-4 py-2 text-gray-100 placeholder-gray-400 focus:outline-none focus:border-blue-400"
                disabled={!file || loading}
              />
              <button
                type="submit"
                disabled={!file || !question.trim() || loading}
                className="bg-blue-500 text-white rounded-lg px-4 py-2 hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Send className="h-5 w-5" />
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
