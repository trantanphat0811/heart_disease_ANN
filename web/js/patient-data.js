/**
 * Patient Data Management Module
 * Handles localStorage operations for patient prediction results
 */

const PatientDataManager = {
  saveBatchResults: function(filename, patientDataWithPredictions, stats) {
    try {
      const batchId = 'batch_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
      const batchRecord = {
        id: batchId,
        batchId: batchId,
        filename: filename,
        uploadDate: new Date().toISOString(),
        totalPatients: patientDataWithPredictions.length,
        patients: patientDataWithPredictions,
        stats: stats
      };
      let history = this.getBatchHistory();
      history.unshift(batchRecord);
      localStorage.setItem('patient_history', JSON.stringify(history));
      let allPatients = this.getAllPatients();
      patientDataWithPredictions.forEach((patient, idx) => {
        allPatients.unshift({
          patientId: 'patient_' + batchId + '_' + idx,
          batchId: batchId,
          ...patient,
          prediction_date: new Date().toISOString()
        });
      });
      localStorage.setItem('all_patients', JSON.stringify(allPatients));
      console.log(`✅ Saved ${patientDataWithPredictions.length} patients from batch ${batchId}`);
      return batchId;
    } catch (error) {
      console.error('Error saving batch results:', error);
      return null;
    }
  },

  getBatchHistory: function() {
    try {
      const data = localStorage.getItem('patient_history');
      return data ? JSON.parse(data) : [];
    } catch (error) {
      console.error('Error getting batch history:', error);
      return [];
    }
  },

  getAllPatients: function() {
    try {
      const data = localStorage.getItem('all_patients');
      return data ? JSON.parse(data) : [];
    } catch (error) {
      console.error('Error getting all patients:', error);
      return [];
    }
  },

  getPatientsByBatch: function(batchId) {
    try {
      const history = this.getBatchHistory();
      const batch = history.find(b => b.id === batchId || b.batchId === batchId);
      return batch ? batch.patients : [];
    } catch (error) {
      console.error('Error getting patients by batch:', error);
      return [];
    }
  },

  getPatientById: function(patientId) {
    try {
      const allPatients = this.getAllPatients();
      return allPatients.find(p => p.patientId === patientId) || null;
    } catch (error) {
      console.error('Error getting patient:', error);
      return null;
    }
  },

  deletePatient: function(patientId) {
    try {
      let allPatients = this.getAllPatients();
      allPatients = allPatients.filter(p => p.patientId !== patientId);
      localStorage.setItem('all_patients', JSON.stringify(allPatients));
      const history = this.getBatchHistory();
      history.forEach(batch => {
        batch.patients = batch.patients.filter(p => p.patientId !== patientId);
        batch.totalPatients = batch.patients.length;
      });
      localStorage.setItem('patient_history', JSON.stringify(history));
      console.log(`✅ Deleted patient ${patientId}`);
      return true;
    } catch (error) {
      console.error('Error deleting patient:', error);
      return false;
    }
  },

  deleteBatch: function(batchId) {
    try {
      let history = this.getBatchHistory();
      history = history.filter(b => b.id !== batchId && b.batchId !== batchId);
      localStorage.setItem('patient_history', JSON.stringify(history));
      let allPatients = this.getAllPatients();
      allPatients = allPatients.filter(p => p.batchId !== batchId);
      localStorage.setItem('all_patients', JSON.stringify(allPatients));
      console.log(`✅ Deleted batch ${batchId}`);
      return true;
    } catch (error) {
      console.error('Error deleting batch:', error);
      return false;
    }
  },

  clearAllData: function() {
    try {
      if (confirm('⚠️ Xác nhận xóa TOÀN BỘ lịch sử bệnh nhân? Hành động này không thể hoàn tác!')) {
        localStorage.removeItem('patient_history');
        localStorage.removeItem('all_patients');
        console.log('✅ Cleared all patient data');
        return true;
      }
      return false;
    } catch (error) {
      console.error('Error clearing data:', error);
      return false;
    }
  },

  exportPatientsToCSV: function(filename) {
    try {
      if (!filename) filename = 'patients_export.csv';
      const allPatients = this.getAllPatients();
      if (allPatients.length === 0) {
        alert('Không có dữ liệu để xuất');
        return false;
      }
      const headers = ['Patient ID', 'Batch ID', 'Age', 'Gender', 'Weight', 'Height', 'BMI', 
                      'Systolic BP', 'Diastolic BP', 'Heart Rate', 'Cholesterol', 'Risk Score', 'Risk Label', 'Date'];
      const csv = [headers.join(',')];
      allPatients.forEach(p => {
        const row = [
          p.patientId || '',
          p.batchId || '',
          p.Age || '',
          p.Gender || '',
          p.Weight || '',
          p.Height || '',
          p.BMI || '',
          p.Systolic_BP || '',
          p.Diastolic_BP || '',
          p.Heart_Rate || '',
          p.Cholesterol_Total || '',
          p.risk_score || '',
          p.label || '',
          p.prediction_date ? new Date(p.prediction_date).toLocaleString('vi-VN') : ''
        ];
        csv.push(row.map(cell => `"${cell}"`).join(','));
      });
      this._downloadFile(csv.join('\n'), filename, 'text/csv');
      console.log(`✅ Exported ${allPatients.length} patients`);
      return true;
    } catch (error) {
      console.error('Error exporting patients:', error);
      return false;
    }
  },

  exportBatchToCSV: function(batchId, filename) {
    try {
      const history = this.getBatchHistory();
      const batch = history.find(b => b.id === batchId || b.batchId === batchId);
      if (!batch) {
        alert('Batch không tìm thấy');
        return false;
      }
      if (!filename) {
        filename = `batch_${batch.filename || batchId}_export.csv`;
      }
      const headers = ['Patient ID', 'Age', 'Gender', 'Weight', 'Height', 'BMI', 
                      'Systolic BP', 'Diastolic BP', 'Risk Score', 'Risk Label'];
      const csv = [headers.join(',')];
      (batch.patients || []).forEach(p => {
        const row = [
          p.patientId || '',
          p.Age || '',
          p.Gender || '',
          p.Weight || '',
          p.Height || '',
          p.BMI || '',
          p.Systolic_BP || '',
          p.Diastolic_BP || '',
          p.risk_score || '',
          p.label || ''
        ];
        csv.push(row.map(cell => `"${cell}"`).join(','));
      });
      this._downloadFile(csv.join('\n'), filename, 'text/csv');
      console.log(`✅ Exported batch with ${batch.patients.length} patients`);
      return true;
    } catch (error) {
      console.error('Error exporting batch:', error);
      return false;
    }
  },

  exportAllDataAsJSON: function(filename) {
    try {
      if (!filename) filename = 'heart_disease_data.json';
      const data = {
        exportDate: new Date().toISOString(),
        summary: this.getStatistics(),
        batches: this.getBatchHistory(),
        patients: this.getAllPatients()
      };
      this._downloadFile(JSON.stringify(data, null, 2), filename, 'application/json');
      console.log('✅ Exported all data');
      return true;
    } catch (error) {
      console.error('Error exporting JSON:', error);
      return false;
    }
  },

  _downloadFile: function(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  },

  filterByRiskLevel: function(riskLevel) {
    try {
      const allPatients = this.getAllPatients();
      if (riskLevel === 'high') {
        return allPatients.filter(p => p.label === 'Cao' || (p.risk_score && p.risk_score > 0.5));
      } else if (riskLevel === 'low') {
        return allPatients.filter(p => p.label === 'Thấp' || (p.risk_score && p.risk_score <= 0.5));
      }
      return allPatients;
    } catch (error) {
      console.error('Error filtering by risk:', error);
      return [];
    }
  },

  getBatchDetails: function(batchId) {
    try {
      const history = this.getBatchHistory();
      return history.find(b => b.id === batchId || b.batchId === batchId) || null;
    } catch (error) {
      console.error('Error getting batch details:', error);
      return null;
    }
  },

  getBatchPatients: function(batchId) {
    try {
      const batch = this.getBatchDetails(batchId);
      return batch ? batch.patients || [] : [];
    } catch (error) {
      console.error('Error getting batch patients:', error);
      return [];
    }
  },

  updatePatient: function(patientId, updates) {
    try {
      let allPatients = this.getAllPatients();
      const index = allPatients.findIndex(p => p.patientId === patientId);
      if (index === -1) return false;
      allPatients[index] = { ...allPatients[index], ...updates };
      localStorage.setItem('all_patients', JSON.stringify(allPatients));
      console.log(`✅ Updated patient ${patientId}`);
      return true;
    } catch (error) {
      console.error('Error updating patient:', error);
      return false;
    }
  },

  searchPatients: function(query) {
    try {
      const allPatients = this.getAllPatients();
      const lowerQuery = query.toLowerCase();
      return allPatients.filter(p => {
        return (p.Age && p.Age.toString().includes(query)) ||
               (p.Gender && p.Gender.toLowerCase().includes(lowerQuery)) ||
               (p.BMI && p.BMI.toString().includes(query)) ||
               (p.patientId && p.patientId.toLowerCase().includes(lowerQuery)) ||
               (p.label && p.label.toLowerCase().includes(lowerQuery));
      });
    } catch (error) {
      console.error('Error searching patients:', error);
      return [];
    }
  },

  getStatistics: function() {
    try {
      const allPatients = this.getAllPatients();
      const history = this.getBatchHistory();
      const highRisk = allPatients.filter(p => p.label === 'Cao' || (p.risk_score && p.risk_score > 0.5)).length;
      const lowRisk = allPatients.filter(p => p.label === 'Thấp' || (p.risk_score && p.risk_score <= 0.5)).length;
      const avgRisk = allPatients.length > 0 
        ? (allPatients.reduce((sum, p) => sum + (p.risk_score || 0), 0) / allPatients.length).toFixed(2)
        : 0;
      return {
        totalBatches: history.length,
        totalPatients: allPatients.length,
        highRisk: highRisk,
        lowRisk: lowRisk,
        highRiskPercent: allPatients.length > 0 ? Math.round((highRisk / allPatients.length) * 100) : 0,
        averageRiskScore: avgRisk,
        lastUpdate: history.length > 0 ? history[0].uploadDate : null
      };
    } catch (error) {
      console.error('Error calculating statistics:', error);
      return {
        totalBatches: 0,
        totalPatients: 0,
        highRisk: 0,
        lowRisk: 0,
        highRiskPercent: 0,
        averageRiskScore: 0,
        lastUpdate: null
      };
    }
  }
};

if (typeof window !== 'undefined') {
  window.PatientDataManager = PatientDataManager;
}
