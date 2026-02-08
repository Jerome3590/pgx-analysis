# PGx Risk Dashboard - Tab Functionality Test Plan

## Summary of Changes

### Issues Fixed:
1. ✅ **Tab Navigation Restructured**: Converted from two-level tab system to single unified navigation
2. ✅ **Separate Tab Content Areas**: Created dedicated content areas for Drugs, ICD Codes, and CPT Codes tabs
3. ✅ **Tab Content Display**: Fixed switchTab function to properly show/hide content
4. ✅ **Drug Selection Display**: Added visual chip display for selected drugs (similar to ICD/CPT)
5. ✅ **Selection Management**: Added remove buttons (×) for each code type with instant updates

### New Tab Structure:
```
Risk Assessment | Drugs | ICD Codes | CPT Codes | Causal Analysis | DTW Trajectories | 
FP-Growth Patterns | BupaR Process Mining | PGx Card | Documentation
```

## Test Procedures

### Test 1: Basic Tab Navigation
**Objective**: Verify all tabs show their content properly

**Steps**:
1. Open `index.html` in a web browser
2. Click on each tab in sequence:
   - Risk Assessment (should be active by default)
   - Drugs
   - ICD Codes
   - CPT Codes
   - Causal Analysis
   - DTW Trajectories
   - FP-Growth Patterns
   - BupaR Process Mining
   - PGx Card
   - Documentation

**Expected Results**:
- ✓ Each tab button becomes highlighted (blue border) when clicked
- ✓ Only one tab's content is visible at a time
- ✓ Previous tab content is hidden when switching to new tab
- ✓ All tabs show their respective content (no blank screens)

**How to Verify**:
- Look for the subtitle text specific to each tab
- Confirm content changes when clicking different tabs
- Check that tab button has blue border when active

---

### Test 2: Drug Selection Display
**Objective**: Verify drug selection works and displays selected drugs

**Steps**:
1. Set age to 35 on Risk Assessment tab (to load metadata)
2. Click "Calculate Risk Score" button to initialize metadata
3. Navigate to **Drugs** tab
4. Use search box to filter drugs (e.g., type "aspirin")
5. Select 2-3 drugs from the multi-select list (Ctrl+click or Cmd+click)
6. Check "Selected Drugs" display area below the list

**Expected Results**:
- ✓ Search box filters the drug list in real-time
- ✓ Selected drugs appear as colored chips below the list
- ✓ Each chip shows drug name and an × button
- ✓ Clicking × removes that drug from selection
- ✓ Display updates immediately when adding/removing drugs

---

### Test 3: ICD Codes Selection Display
**Objective**: Verify ICD code selection works (for Opioid ED patients)

**Steps**:
1. Ensure age is set to a value between 13-64 on Risk Assessment tab
2. Navigate to **ICD Codes** tab
3. Use search box to filter ICD codes
4. Select 2-3 ICD codes from the list
5. Check "Selected ICD Codes" display area

**Expected Results**:
- ✓ ICD Codes tab is visible for ages 13-64
- ✓ Search box filters ICD codes in real-time
- ✓ Selected ICD codes appear as colored chips
- ✓ Each chip has an × button that removes the code
- ✓ Display updates immediately

---

### Test 4: CPT Codes Selection Display
**Objective**: Verify CPT code selection works (for Opioid ED patients)

**Steps**:
1. Ensure age is set to a value between 13-64 on Risk Assessment tab
2. Navigate to **CPT Codes** tab
3. Use search box to filter CPT codes
4. Select 2-3 CPT codes from the list
5. Check "Selected CPT Codes" display area

**Expected Results**:
- ✓ CPT Codes tab is visible for ages 13-64
- ✓ Search box filters CPT codes in real-time
- ✓ Selected CPT codes appear as colored chips
- ✓ Each chip has an × button that removes the code
- ✓ Display updates immediately

---

### Test 5: Polypharmacy Patient (ICD/CPT Hidden)
**Objective**: Verify ICD and CPT tabs are hidden for polypharmacy patients

**Steps**:
1. Go to **Risk Assessment** tab
2. Change age to 70 (polypharmacy range: 65-114)
3. Wait a moment for the interface to update
4. Check which tabs are visible in the navigation

**Expected Results**:
- ✓ ICD Codes tab button is hidden
- ✓ CPT Codes tab button is hidden
- ✓ Drugs tab is still visible and functional
- ✓ If currently on ICD/CPT tab when age changes, automatically switches to Drugs tab

---

### Test 6: Risk Calculation with Codes
**Objective**: Test that selected codes are used in risk calculation

**Steps**:
1. Go to **Risk Assessment** tab
2. Set age to 35
3. Navigate to **Drugs** tab and select 2-3 drugs
4. Navigate to **ICD Codes** tab and select 1-2 codes
5. Navigate to **CPT Codes** tab and select 1-2 codes
6. Return to **Risk Assessment** tab
7. Check the codes summary section (should show count of selected codes)
8. Click "Calculate Risk Score"

**Expected Results**:
- ✓ Summary shows correct count: "Selected: X drug(s), Y ICD(s), Z CPT(s)"
- ✓ "Edit codes" button navigates to Drugs tab when clicked
- ✓ Risk calculation completes successfully
- ✓ Risk score is displayed with appropriate risk band

---

### Test 7: Selection Persistence
**Objective**: Verify selections persist when navigating between tabs

**Steps**:
1. Select drugs on Drugs tab
2. Select ICD codes on ICD Codes tab
3. Select CPT codes on CPT Codes tab
4. Navigate back to Drugs tab
5. Check if previously selected drugs are still shown in chips

**Expected Results**:
- ✓ Drug selections persist and remain visible
- ✓ ICD selections persist when returning to ICD tab
- ✓ CPT selections persist when returning to CPT tab
- ✓ All selections remain until explicitly removed

---

### Test 8: Visualization Tabs
**Objective**: Verify visualization tabs are accessible and functional

**Steps**:
1. Navigate to **Causal Analysis** tab
2. Select cohort and age band
3. Click "Load Causal Analysis"
4. Repeat for **DTW Trajectories**, **FP-Growth Patterns**, and **BupaR Process Mining** tabs

**Expected Results**:
- ✓ Each visualization tab shows its controls
- ✓ Dropdowns and buttons are visible and functional
- ✓ Tab content is properly displayed (no blank screens)
- ✓ Status messages appear when loading visualizations

---

### Test 9: PGx Card Tab
**Objective**: Verify PGx Patient Card tab works

**Steps**:
1. Navigate to **PGx Card** tab
2. Verify form elements are visible:
   - Patient ID input (optional)
   - SNP Data textarea
   - File upload button
   - Generate and Reset buttons

**Expected Results**:
- ✓ Tab content is visible
- ✓ All form elements are present
- ✓ Instructions and privacy note are displayed

---

### Test 10: Documentation Tab
**Objective**: Verify documentation is updated and accessible

**Steps**:
1. Navigate to **Documentation** tab
2. Read through the content

**Expected Results**:
- ✓ Documentation reflects new tab structure
- ✓ Instructions mention separate Drugs, ICD, CPT tabs
- ✓ Workflow section is accurate
- ✓ All sections are readable and properly formatted

---

## Known Issues to Watch For

### Issue 1: Initial Tab Not Showing
**Symptoms**: When page first loads, Risk Assessment tab content is blank
**Fix**: Refresh the page, or check browser console for JavaScript errors

### Issue 2: Selections Not Displaying
**Symptoms**: Chips for selected codes don't appear
**Fix**: Open browser developer tools (F12) and check console for errors

### Issue 3: ICD/CPT Not Hiding for Polypharmacy
**Symptoms**: ICD/CPT tabs still visible when age is 65+
**Fix**: Click away from age field or press Enter after entering age

---

## Browser Testing

Test in multiple browsers to ensure compatibility:
- [ ] Chrome/Edge (Chromium)
- [ ] Firefox
- [ ] Safari (if on Mac)

---

## Success Criteria

All tests should pass with the following results:
1. ✅ All 10 tabs show content (no blank screens)
2. ✅ Drug selection display shows chips with remove buttons
3. ✅ ICD selection display shows chips with remove buttons
4. ✅ CPT selection display shows chips with remove buttons
5. ✅ Tab navigation is smooth with proper highlighting
6. ✅ Selections persist across tab switches
7. ✅ Risk calculation uses selected codes
8. ✅ ICD/CPT tabs hidden for polypharmacy patients

---

## Troubleshooting

### If tabs are not switching:
1. Open browser developer tools (F12)
2. Go to Console tab
3. Look for JavaScript errors
4. Report any errors with full stack trace

### If selection displays are blank:
1. Check if drugs/ICD/CPT elements exist in HTML
2. Look for `selected-drugs-display`, `selected-icds-display`, `selected-cpts-display` IDs
3. Verify event listeners are attached (check Console for errors)

### If metadata doesn't load:
1. Check API_BASE_URL is correct
2. Verify network connection
3. Check browser Network tab for failed requests
4. Ensure backend API is running

---

## Comparison with PHTS Template

The updated PGx dashboard now follows the PHTS pattern:
- ✅ Single-level tab navigation (vs. two-level)
- ✅ Each tab has dedicated content area
- ✅ Clean tab switching with one active tab at a time
- ✅ Visual feedback for selections (chips with remove buttons)
- ✅ Proper content visibility management

---

## Next Steps After Testing

1. **If all tests pass**: Dashboard is ready for production use
2. **If tests fail**: 
   - Document specific failures
   - Include browser console errors
   - Provide screenshots if possible
   - Note which test number failed

3. **Future enhancements**:
   - Add drug-drug interaction analysis (separate feature)
   - Implement advanced filtering for code selection
   - Add export functionality for selected codes
   - Consider adding tooltips for code descriptions
