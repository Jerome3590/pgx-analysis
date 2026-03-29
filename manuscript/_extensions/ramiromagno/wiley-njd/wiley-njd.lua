
-- Convert `# Acknowledgments {.bmsection}` to `\bmsection*{Acknowledgments}`
--
-- this function will be executed for all Header elements in the document
Header = function(el)
  -- el.level == 1 makes it so you only catch "# ..." and not "## ...", etc
  -- el.classes is a pandoc List object, which has the method "includes"
  local bm_cmd = {
    [1] = "\\bmsection",
    [2] = "\\bmsubsection",
    [3] = "\\bmsubsubsection"
  }
  
  if el.classes:includes("bm") then
    if el.classes:includes("unnumbered") then
      -- pandoc.utils.stringify converts a pandoc value (in this case,
      -- the list of inlines that make up the header) into a plain string.
      local text = pandoc.utils.stringify(el.content)

      -- a RawBlock element contains a string that is passed through
      -- directly to the final element.
      local result = pandoc.RawBlock("latex", "" .. bm_cmd[el.level] .. "*{" .. text .. "}")
      return result
    else
      -- pandoc.utils.stringify converts a pandoc value (in this case,
      -- the list of inlines that make up the header) into a plain string.
      local text = pandoc.utils.stringify(el.content)

      -- a RawBlock element contains a string that is passed through
      -- directly to the final element.
      local result = pandoc.RawBlock("latex", "" .. bm_cmd[el.level] .. "{" .. text .. "}")
      return result
    end
  else
    -- a "return nil" statement is implied at the end of a lua function, but
    -- if you want to be explicit, you can add it.
    -- returning "nil" means "do not modify this element"
    return nil
  end
end


-- Quarto cross-reference prefixes — do NOT convert these to \cite{}
local xref_prefixes = {
  "fig%-", "tbl%-", "sec%-", "eq%-", "lst%-",
  "thm%-", "lem%-", "cor%-", "prp%-", "def%-", "exm%-", "exr%-", "rem%-"
}

local function is_xref(id)
  for _, prefix in ipairs(xref_prefixes) do
    if id:match("^" .. prefix) then return true end
  end
  return false
end

Cite = function(el)
  if quarto.doc.is_format("pdf") then
    -- Filter out cross-reference IDs; keep only real bibliography keys
    local bib_keys = {}
    for _, cite in ipairs(el.citations) do
      if not is_xref(cite.id) then
        table.insert(bib_keys, cite.id)
      end
    end
    if #bib_keys == 0 then
      -- All were cross-refs — return nil so Quarto handles them natively
      return nil
    end
    local citesStr = "\\cite{" .. table.concat(bib_keys, ",") .. "}"
    return pandoc.RawInline("latex", citesStr)
  end
end

